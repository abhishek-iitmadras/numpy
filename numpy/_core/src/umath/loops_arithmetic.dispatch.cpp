// Copyright (c) 2024, NumPy Developers.
#define _UMATHMODULE
#define _MULTIARRAYMODULE
#define NPY_NO_DEPRECATED_API NPY_API_VERSION

#include "loops_utils.h"
#include "loops.h"
#include <cstring> // for memcpy
#include "fast_loop_macros.h"
#include <limits>
#include "simd/simd.h"
#include "lowlevel_strided_loops.h"
#include "numpy/npy_math.h"
#include <cstdio>

#include <hwy/highway.h>
namespace hn = hwy::HWY_NAMESPACE;

HWY_BEFORE_NAMESPACE();
namespace HWY_NAMESPACE {

// Helper struct for tracking division errors
struct DivisionState {
    bool overflow;
    bool divbyzero;
    
    DivisionState() : overflow(false), divbyzero(false) {}
    
    void update(bool of, bool dz) {
        overflow |= of;
        divbyzero |= dz;
    }
    
    void set_status() const {
        if (overflow) {
            npy_set_floatstatus_overflow();
        }
        if (divbyzero) {
            npy_set_floatstatus_divbyzero();
        }
    }
};

// Helper function to check memory overlap
inline bool check_overlap(const void* dst, const void* src, size_t len) {
    return (static_cast<const char*>(dst) + len <= static_cast<const char*>(src) ||
            static_cast<const char*>(src) + len <= static_cast<const char*>(dst));
}

// Floor division implementation
template <typename T>
inline T floor_div(T n, T d) {
    if (HWY_UNLIKELY(d == 0 || (n == std::numeric_limits<T>::min() && d == -1))) {
        if (d == 0) {
            npy_set_floatstatus_divbyzero();
            return 0;
        }
        else {
            npy_set_floatstatus_overflow();
            return std::numeric_limits<T>::min();
        }
    }
    T q = n / d;
    T r = n % d;
    if (r != 0 && ((r < 0) != (d < 0))) {
        q--;
    }
    return q;
}

// SIMD implementation for signed integer division
template <typename T>
void simd_divide_by_scalar_contig_signed(T* src, T scalar, T* dst, npy_intp len) {
    using D = hn::ScalableTag<T>;
    const D d;
    const size_t N = hn::Lanes(d);
    
    DivisionState state;

    // Check for memory overlap
    if (check_overlap(dst, src, len * sizeof(T))) {
        if (scalar == 0) {
            std::fill(dst, dst + len, static_cast<T>(0));
            state.update(false, true);
        }
        else if (scalar == 1) {
            memcpy(dst, src, len * sizeof(T));
        }
        else if (scalar == -1) {
            const auto vec_min_val = hn::Set(d, std::numeric_limits<T>::min());
            const auto zero = hn::Zero(d);
            
            size_t i = 0;
            for (; i + N <= static_cast<size_t>(len); i += N) {
                const auto vec_src = hn::LoadU(d, src + i);
                const auto is_min_val = hn::Eq(vec_src, vec_min_val);
                auto vec_res = hn::Neg(vec_src);
                vec_res = hn::IfThenElse(is_min_val, vec_min_val, vec_res);
                hn::StoreU(vec_res, d, dst + i);
                state.update(!hn::AllFalse(d, is_min_val), false);
            }
            
            for (; i < static_cast<size_t>(len); i++) {
                if (src[i] == std::numeric_limits<T>::min()) {
                    dst[i] = std::numeric_limits<T>::min();
                    state.update(true, false);
                } else {
                    dst[i] = -src[i];
                }
            }
        }
        else {
            const auto vec_scalar = hn::Set(d, scalar);
            const auto zero = hn::Zero(d);
            const auto one = hn::Set(d, static_cast<T>(1));
            
            size_t i = 0;
            for (; i + N <= static_cast<size_t>(len); i += N) {
                const auto vec_src = hn::LoadU(d, src + i);
                auto vec_res = hn::Div(vec_src, vec_scalar);
                
                // Implement floor division logic
                const auto vec_mul = hn::Mul(vec_res, vec_scalar);
                const auto has_remainder = hn::Ne(vec_src, vec_mul);
                const auto vec_nsign_src = hn::Lt(vec_src, zero);
                const auto vec_nsign_scalar = hn::Lt(vec_scalar, zero);
                const auto needs_adjustment = hn::And(
                    has_remainder,
                    hn::Xor(vec_nsign_src, vec_nsign_scalar)
                );
                vec_res = hn::IfThenElse(needs_adjustment, 
                                       hn::Sub(vec_res, one),
                                       vec_res);
                hn::StoreU(vec_res, d, dst + i);
            }
            
            for (; i < static_cast<size_t>(len); i++) {
                dst[i] = floor_div(src[i], scalar);
            }
        }
    } else {
        // Handle overlapping memory with scalar operations
        for (size_t i = 0; i < static_cast<size_t>(len); i++) {
            dst[i] = floor_div(src[i], scalar);
        }
    }
    
    state.set_status();
}

// SIMD implementation for unsigned integer division
template <typename T>
void simd_divide_by_scalar_contig_unsigned(T* src, T scalar, T* dst, npy_intp len) {
    using D = hn::ScalableTag<T>;
    const D d;
    const size_t N = hn::Lanes(d);
    
    DivisionState state;

    if (check_overlap(dst, src, len * sizeof(T))) {
        if (scalar == 0) {
            std::fill(dst, dst + len, static_cast<T>(0));
            state.update(false, true);
        }
        else if (scalar == 1) {
            memcpy(dst, src, len * sizeof(T));
        }
        else {
            const auto vec_scalar = hn::Set(d, scalar);
            
            size_t i = 0;
            for (; i + N <= static_cast<size_t>(len); i += N) {
                const auto vec_src = hn::LoadU(d, src + i);
                const auto vec_res = hn::Div(vec_src, vec_scalar);
                hn::StoreU(vec_res, d, dst + i);
            }
            
            for (; i < static_cast<size_t>(len); i++) {
                dst[i] = src[i] / scalar;
            }
        }
    } else {
        // Handle overlapping memory with scalar operations
        for (size_t i = 0; i < static_cast<size_t>(len); i++) {
            if (scalar == 0) {
                dst[i] = 0;
                state.update(false, true);
            } else {
                dst[i] = src[i] / scalar;
            }
        }
    }
    
    state.set_status();
}

// Dispatch function for signed integer division
template <typename T>
void TYPE_divide(char **args, npy_intp const *dimensions, npy_intp const *steps, void *NPY_UNUSED(func)) {
    // Handle binary reduction case
    if (IS_BINARY_REDUCE) {
        T io1 = *reinterpret_cast<T*>(args[0]);
        char *ip2 = args[1];
        npy_intp is2 = steps[1];
        
        for (npy_intp i = 0; i < dimensions[0]; i++, ip2 += is2) {
            const T divisor = *reinterpret_cast<T*>(ip2);
            io1 = floor_div(io1, divisor);
        }
        *reinterpret_cast<T*>(args[0]) = io1;
        return;
    }
    
    // Check for in-place operation
    if (args[0] == args[2] || args[1] == args[2]) {
        BINARY_LOOP {
            const T dividend = *reinterpret_cast<T*>(ip1);
            const T divisor = *reinterpret_cast<T*>(ip2);
            *reinterpret_cast<T*>(op1) = floor_div(dividend, divisor);
        }
        return;
    }
    
    // Check if we can use SIMD
    if (IS_BLOCKABLE_BINARY_SCALAR2(sizeof(T), NPY_SIMD_WIDTH) &&
        *reinterpret_cast<T*>(args[1]) != 0)
    {
        bool no_overlap = nomemoverlap(args[2], steps[2], args[0], steps[0], dimensions[0]);
        if (no_overlap) {
            T* src1 = reinterpret_cast<T*>(args[0]);
            T* src2 = reinterpret_cast<T*>(args[1]);
            T* dst = reinterpret_cast<T*>(args[2]);
            simd_divide_by_scalar_contig_signed(src1, *src2, dst, dimensions[0]);
            return;
        }
    }
    
    // Fallback to scalar operations
    BINARY_LOOP {
        const T dividend = *reinterpret_cast<T*>(ip1);
        const T divisor = *reinterpret_cast<T*>(ip2);
        *reinterpret_cast<T*>(op1) = floor_div(dividend, divisor);
    }
}

// Dispatch function for unsigned integer division
template <typename T>
void TYPE_divide_unsigned(char **args, npy_intp const *dimensions, npy_intp const *steps, void *NPY_UNUSED(func)) {
    if (IS_BINARY_REDUCE) {
        T io1 = *reinterpret_cast<T*>(args[0]);
        char *ip2 = args[1];
        npy_intp is2 = steps[1];
        
        for (npy_intp i = 0; i < dimensions[0]; i++, ip2 += is2) {
            const T divisor = *reinterpret_cast<T*>(ip2);
            if (HWY_UNLIKELY(divisor == 0)) {
                npy_set_floatstatus_divbyzero();
                io1 = 0;
            } else {
                io1 = io1 / divisor;
            }
        }
        *reinterpret_cast<T*>(args[0]) = io1;
        return;
    }
    
    if (args[0] == args[2] || args[1] == args[2]) {
        BINARY_LOOP {
            const T in1 = *reinterpret_cast<T*>(ip1);
            const T in2 = *reinterpret_cast<T*>(ip2);
            if (HWY_UNLIKELY(in2 == 0)) {
                npy_set_floatstatus_divbyzero();
                *reinterpret_cast<T*>(op1) = 0;
            } else {
                *reinterpret_cast<T*>(op1) = in1 / in2;
            }
        }
        return;
    }
    
    if (IS_BLOCKABLE_BINARY_SCALAR2(sizeof(T), NPY_SIMD_WIDTH) &&
        *reinterpret_cast<T*>(args[1]) != 0)
    {
        bool no_overlap = nomemoverlap(args[2], steps[2], args[0], steps[0], dimensions[0]);
        if (no_overlap) {
            T* src1 = reinterpret_cast<T*>(args[0]);
            T* src2 = reinterpret_cast<T*>(args[1]);
            T* dst = reinterpret_cast<T*>(args[2]);
            simd_divide_by_scalar_contig_unsigned(src1, *src2, dst, dimensions[0]);
            return;
        }
    }
    
    BINARY_LOOP {
        const T in1 = *reinterpret_cast<T*>(ip1);
        const T in2 = *reinterpret_cast<T*>(ip2);
        if (HWY_UNLIKELY(in2 == 0)) {
            npy_set_floatstatus_divbyzero();
            *reinterpret_cast<T*>(op1) = 0;
        } else {
            *reinterpret_cast<T*>(op1) = in1 / in2;
        }
    }
}

// Indexed division for signed integers
template <typename T>
int TYPE_divide_indexed(char * const*args, npy_intp const *dimensions, 
                       npy_intp const *steps, NpyAuxData *NPY_UNUSED(func)) {
    char *ip1 = args[0];
    char *indxp = args[1];
    char *value = args[2];
    npy_intp is1 = steps[0], isindex = steps[1], isb = steps[2];
    npy_intp shape = steps[3];
    npy_intp n = dimensions[0];
    
    for(npy_intp i = 0; i < n; i++, indxp += isindex, value += isb) {
        npy_intp indx = *(npy_intp *)indxp;
        if (indx < 0) {
            indx += shape;
        }
        T* indexed = reinterpret_cast<T*>(ip1 + is1 * indx);
        T divisor = *reinterpret_cast<T*>(value);
        *indexed = floor_div(*indexed, divisor);
    }
    return 0;
}

// Indexed division for unsigned integers
template <typename T>
int TYPE_divide_unsigned_indexed(char * const*args, npy_intp const *dimensions, 
                               npy_intp const *steps, NpyAuxData *NPY_UNUSED(func)) {
    char *ip1 = args[0];
    char *indxp = args[1];
    char *value = args[2];
    npy_intp is1 = steps[0], isindex = steps[1], isb = steps[2];
    npy_intp shape = steps[3];
    npy_intp n = dimensions[0];
    
    for(npy_intp i = 0; i < n; i++, indxp += isindex, value += isb) {
        npy_intp indx = *(npy_intp *)indxp;
        if (indx < 0) {
            indx += shape;
        }
        T* indexed = reinterpret_cast<T*>(ip1 + is1 * indx);
        T divisor = *reinterpret_cast<T*>(value);
        
        if (HWY_UNLIKELY(divisor == 0)) {
            npy_set_floatstatus_divbyzero();
            *indexed = 0;
        } else {
            *indexed = *indexed / divisor;
        }
    }
    return 0;
}

#define DEFINE_DIVIDE_FUNCTION(TYPE, SCALAR_TYPE) \
    extern "C" { \
        NPY_NO_EXPORT void NPY_CPU_DISPATCH_CURFX(TYPE##_divide)(char **args, npy_intp const *dimensions, npy_intp const *steps, void *func) { \
            TYPE_divide<SCALAR_TYPE>(args, dimensions, steps, func); \
        } \
        NPY_NO_EXPORT int NPY_CPU_DISPATCH_CURFX(TYPE##_divide_indexed)(PyArrayMethod_Context *context, char * const*args, npy_intp const *dimensions, npy_intp const *steps, NpyAuxData *func) { \
            return TYPE_divide_indexed<SCALAR_TYPE>(args, dimensions, steps, func); \
        } \
    }

#define DEFINE_DIVIDE_FUNCTION_UNSIGNED(TYPE, SCALAR_TYPE) \
    extern "C" { \
        NPY_NO_EXPORT void NPY_CPU_DISPATCH_CURFX(TYPE##_divide)(char **args, npy_intp const *dimensions, npy_intp const *steps, void *func) { \
            TYPE_divide_unsigned<SCALAR_TYPE>(args, dimensions, steps, func); \
        } \
        NPY_NO_EXPORT int NPY_CPU_DISPATCH_CURFX(TYPE##_divide_indexed)(PyArrayMethod_Context *context, char * const*args, npy_intp const *dimensions, npy_intp const *steps, NpyAuxData *func) { \
            return TYPE_divide_unsigned_indexed<SCALAR_TYPE>(args, dimensions, steps, func); \
        } \
    }

#ifdef NPY_CPU_DISPATCH_CURFX
DEFINE_DIVIDE_FUNCTION(BYTE, int8_t)
DEFINE_DIVIDE_FUNCTION(SHORT, int16_t)
DEFINE_DIVIDE_FUNCTION(INT, int32_t)
DEFINE_DIVIDE_FUNCTION(LONG, int64_t)
DEFINE_DIVIDE_FUNCTION(LONGLONG, int64_t)

DEFINE_DIVIDE_FUNCTION_UNSIGNED(UBYTE, uint8_t)
DEFINE_DIVIDE_FUNCTION_UNSIGNED(USHORT, uint16_t)
DEFINE_DIVIDE_FUNCTION_UNSIGNED(UINT, uint32_t)
DEFINE_DIVIDE_FUNCTION_UNSIGNED(ULONG, uint64_t)
DEFINE_DIVIDE_FUNCTION_UNSIGNED(ULONGLONG, uint64_t)
#endif

#undef DEFINE_DIVIDE_FUNCTION
#undef DEFINE_DIVIDE_FUNCTION_UNSIGNED

} // namespace HWY_NAMESPACE
HWY_AFTER_NAMESPACE();