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

// Helper function to set float status
inline void set_float_status(bool overflow, bool divbyzero) {
    if (overflow) {
        npy_set_floatstatus_overflow();
    }
    if (divbyzero) {
        npy_set_floatstatus_divbyzero();
    }
}

// Signed integer division
template <typename T>
void simd_divide_by_scalar_contig_signed(T* src, T scalar, T* dst, npy_intp len) {
    using D = hn::ScalableTag<T>;
    const D d;
    const size_t N = hn::Lanes(d);

    bool raise_overflow = false;
    bool raise_divbyzero = false;

    if (scalar == 0) {
        // Handle division by zero
        std::fill(dst, dst + len, static_cast<T>(0));
        raise_divbyzero = true;
    }
    else if (scalar == 1) {
        // Special case for division by 1
        memcpy(dst, src, len * sizeof(T));
    }
    else if (scalar == static_cast<T>(-1)) {
        const auto vec_min_val = hn::Set(d, std::numeric_limits<T>::min());
        size_t i = 0;
        for (; i + N <= static_cast<size_t>(len); i += N) {
            const auto vec_src = hn::LoadU(d, src + i);
            const auto is_min_val = hn::Eq(vec_src, vec_min_val);
            const auto vec_res = hn::IfThenElse(is_min_val, vec_min_val, hn::Neg(vec_src));
            hn::StoreU(vec_res, d, dst + i);
            if (!raise_overflow && !hn::AllFalse(d, is_min_val)) {
                raise_overflow = true;
            }
        }
        if (i < static_cast<size_t>(len)) {
            const size_t num = len - i;
            const auto vec_src = hn::LoadN(d, src + i, num);
            const auto is_min_val = hn::Eq(vec_src, vec_min_val);
            const auto vec_res = hn::IfThenElse(is_min_val, vec_min_val, hn::Neg(vec_src));
            hn::StoreN(vec_res, d, dst + i, num);
            if (!raise_overflow && !hn::AllFalse(d, is_min_val)) {
                raise_overflow = true;
            }
        }
    }
    else {
        const auto vec_scalar = hn::Set(d, scalar);
        const auto zero = hn::Zero(d);
        size_t i = 0;
        for (; i + N <= static_cast<size_t>(len); i += N) {
            const auto vec_src = hn::LoadU(d, src + i);
            auto vec_res = hn::Div(vec_src, vec_scalar);
            const auto vec_mul = hn::Mul(vec_res, vec_scalar);
            const auto remainder_check = hn::Ne(vec_src, vec_mul);
            const auto vec_nsign_src = hn::Lt(vec_src, zero);
            const auto vec_nsign_scalar = hn::Lt(vec_scalar, zero);
            const auto diff_sign = hn::Xor(vec_nsign_src, vec_nsign_scalar);
            vec_res = hn::IfThenElse(
                hn::And(remainder_check, diff_sign),
                hn::Sub(vec_res, hn::Set(d, 1)),
                vec_res
            );
            hn::StoreU(vec_res, d, dst + i);
        }
        if (i < static_cast<size_t>(len)) {
            const size_t num = len - i;
            const auto vec_src = hn::LoadN(d, src + i, num);
            auto vec_res = hn::Div(vec_src, vec_scalar);
            const auto vec_mul = hn::Mul(vec_res, vec_scalar);
            const auto remainder_check = hn::Ne(vec_src, vec_mul);
            const auto vec_nsign_src = hn::Lt(vec_src, zero);
            const auto vec_nsign_scalar = hn::Lt(vec_scalar, zero);
            const auto diff_sign = hn::Xor(vec_nsign_src, vec_nsign_scalar);
            vec_res = hn::IfThenElse(
                hn::And(remainder_check, diff_sign),
                hn::Sub(vec_res, hn::Set(d, 1)),
                vec_res
            );
            hn::StoreN(vec_res, d, dst + i, num);
        }
    }

    set_float_status(raise_overflow, raise_divbyzero);
}

// Unsigned integer division
template <typename T>
void simd_divide_by_scalar_contig_unsigned(T* src, T scalar, T* dst, npy_intp len) {
    using D = hn::ScalableTag<T>;
    const D d;
    const size_t N = hn::Lanes(d);

    bool raise_divbyzero = false;

    if (scalar == 0) {
        // Handle division by zero
        std::fill(dst, dst + len, static_cast<T>(0));
        raise_divbyzero = true;
    }
    else if (scalar == 1) {
        // Special case for division by 1
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
        if (i < static_cast<size_t>(len)) {
            const size_t num = len - i;
            const auto vec_src = hn::LoadN(d, src + i, num);
            const auto vec_res = hn::Div(vec_src, vec_scalar);
            hn::StoreN(vec_res, d, dst + i, num);
        }
    }

    set_float_status(false, raise_divbyzero);
}

// Floor division for signed integers
template <typename T>
T floor_div(T n, T d) {
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
    T r = n / d;
    if (((n > 0) != (d > 0)) && ((r * d) != n)) {
        r--;
    }
    return r;
}

// Dispatch functions for signed integer division
template <typename T>
void TYPE_divide(char **args, npy_intp const *dimensions, npy_intp const *steps, void *NPY_UNUSED(func)) {
    if (IS_BINARY_REDUCE) {
        BINARY_REDUCE_LOOP(T) {
            const T divisor = *reinterpret_cast<T*>(ip2);
            if (HWY_UNLIKELY(divisor == 0)) {
                npy_set_floatstatus_divbyzero();
                io1 = 0;
            } else if (HWY_UNLIKELY(io1 == std::numeric_limits<T>::min() && divisor == -1)) {
                npy_set_floatstatus_overflow();
                io1 = std::numeric_limits<T>::min();
            } else {
                io1 = floor_div(io1, divisor);
            }
        }
        *reinterpret_cast<T*>(iop1) = io1;
    }
    else if (steps[0] == sizeof(T) && steps[1] == 0 && steps[2] == sizeof(T)) {
        T* src1 = reinterpret_cast<T*>(args[0]);
        T* src2 = reinterpret_cast<T*>(args[1]);
        T* dst = reinterpret_cast<T*>(args[2]);
        
        if (HWY_UNLIKELY(*src2 == 0)) {
            npy_set_floatstatus_divbyzero();
            std::fill(dst, dst + dimensions[0], 0);
        } else {
            simd_divide_by_scalar_contig_signed(src1, *src2, dst, dimensions[0]);
        }
    }
    else {
        BINARY_LOOP {
            const T dividend = *reinterpret_cast<T*>(ip1);
            const T divisor = *reinterpret_cast<T*>(ip2);
            T* result = reinterpret_cast<T*>(op1);
            
            if (HWY_UNLIKELY(divisor == 0)) {
                npy_set_floatstatus_divbyzero();
                *result = 0;
            } else if (HWY_UNLIKELY(dividend == std::numeric_limits<T>::min() && divisor == -1)) {
                npy_set_floatstatus_overflow();
                *result = std::numeric_limits<T>::min();
            } else {
                *result = floor_div(dividend, divisor);
            }
        }
    }
}

// Dispatch functions for unsigned integer division
template <typename T>
void TYPE_divide_unsigned(char **args, npy_intp const *dimensions, npy_intp const *steps, void *NPY_UNUSED(func)) {
    if (IS_BINARY_REDUCE) {
        BINARY_REDUCE_LOOP(T) {
            const T d = *reinterpret_cast<T*>(ip2);
            if (HWY_UNLIKELY(d == 0)) {
                npy_set_floatstatus_divbyzero();
                io1 = 0;
            } else {
                io1 /= d;
            }
        }
        *reinterpret_cast<T*>(iop1) = io1;
    }
    else if (steps[0] == sizeof(T) && steps[1] == 0 && steps[2] == sizeof(T)) {
        T* src1 = reinterpret_cast<T*>(args[0]);
        T* src2 = reinterpret_cast<T*>(args[1]);
        T* dst = reinterpret_cast<T*>(args[2]);
        
        if (HWY_UNLIKELY(*src2 == 0)) {
            npy_set_floatstatus_divbyzero();
            std::fill(dst, dst + dimensions[0], 0);
        } else {
            simd_divide_by_scalar_contig_unsigned(src1, *src2, dst, dimensions[0]);
        }
    }
    else {
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
}

// Indexed division for signed integers
template <typename T>
int TYPE_divide_indexed(char * const*args, npy_intp const *dimensions, npy_intp const *steps, NpyAuxData *NPY_UNUSED(func)) {
    char *ip1 = args[0];
    char *indxp = args[1];
    char *value = args[2];
    npy_intp is1 = steps[0], isindex = steps[1], isb = steps[2];
    npy_intp shape = steps[3];
    npy_intp n = dimensions[0];
    npy_intp i;
    T *indexed;
    for(i = 0; i < n; i++, indxp += isindex, value += isb) {
        npy_intp indx = *(npy_intp *)indxp;
        if (indx < 0) {
            indx += shape;
        }
        indexed = (T *)(ip1 + is1 * indx);
        *indexed = floor_div(*indexed, *(T *)value);
    }
    return 0;
}

// Indexed division for unsigned integers
template <typename T>
int TYPE_divide_unsigned_indexed(char * const*args, npy_intp const *dimensions, npy_intp const *steps, NpyAuxData *NPY_UNUSED(func)) {
    char *ip1 = args[0];
    char *indxp = args[1];
    char *value = args[2];
    npy_intp is1 = steps[0], isindex = steps[1], isb = steps[2];
    npy_intp shape = steps[3];
    npy_intp n = dimensions[0];
    npy_intp i;
    T *indexed;
    for(i = 0; i < n; i++, indxp += isindex, value += isb) {
        npy_intp indx = *(npy_intp *)indxp;
        if (indx < 0) {
            indx += shape;
        }
        indexed = (T *)(ip1 + is1 * indx);
        T in2 = *(T *)value;
        if (HWY_UNLIKELY(in2 == 0)) {
            npy_set_floatstatus_divbyzero();
            *indexed = 0;
        } else {
            *indexed = *indexed / in2;
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
    } // extern "C"


#ifdef NPY_CPU_DISPATCH_CURFX
DEFINE_DIVIDE_FUNCTION(BYTE, int8_t)
DEFINE_DIVIDE_FUNCTION(SHORT, int16_t)
DEFINE_DIVIDE_FUNCTION(INT, int32_t)
DEFINE_DIVIDE_FUNCTION(LONG, int64_t)
DEFINE_DIVIDE_FUNCTION(LONGLONG, int64_t)
#endif

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