#define _UMATHMODULE
#define _MULTIARRAYMODULE
#define NPY_NO_DEPRECATED_API NPY_API_VERSION

#include "hwy/highway.h"
#include "loops_utils.h"
#include "loops.h"
#include "fast_loop_macros.h"
#include <limits>
#include "simd/simd.h"
#include "lowlevel_strided_loops.h"
#include <cstdio>  // For std::fprintf and std::abort

namespace hn = hwy::HWY_NAMESPACE;

#ifndef NPY_CPU_DISPATCH_CURFX
    #define NPY_CPU_DISPATCH_CURFX(FN) FN##_AVX2, FN##_SSE41, FN##_SSE2, FN##_NEON, FN##_BASELINE
#endif


// Macro to map SIMD suffix to Google Highway functions
#if NPY_BITSOF_TYPE == 32
    #define TO_SIMD_SFX(X) X##_int32
#elif NPY_BITSOF_TYPE == 64
    #define TO_SIMD_SFX(X) X##_int64
#endif

// Handle 64-bit division disabling for specific architectures
#if (defined(NPY_HAVE_VSX) && !defined(NPY_HAVE_VSX4)) || defined(NPY_HAVE_NEON)
    #define SIMD_DISABLE_DIV64_OPT
#endif

/********************************************************************************
 ** SIMD Cleanup
 ********************************************************************************/
#if NPY_SIMD
    #define SIMD_CLEANUP() npyv_cleanup()
#else
    #define SIMD_CLEANUP() 
#endif

template <typename T>
T floor_div(T n, T d) {
    if (d == 0) {
        npy_set_floatstatus_divbyzero();
        return 0;
    }
    if (std::numeric_limits<T>::is_signed && n == static_cast<T>(std::numeric_limits<T>::min()) && d == static_cast<T>(-1)) {
        npy_set_floatstatus_overflow();
        return std::numeric_limits<T>::min();
    }
    T r = n / d;
    if (std::numeric_limits<T>::is_signed && ((n > 0) != (d > 0)) && (r * d != n)) {
        r--;
    }
    return r;
}

template <typename T>
void simd_divide_by_scalar_contig_signed(T* src, T scalar, T* dst, npy_intp len, hn::ScalableTag<T> tag) {
    const int lanes = hn::Lanes(tag);
    // Precompute the reciprocal (1 / scalar) for optimized division
    auto vec_reciprocal = hn::Set(tag, static_cast<T>(1) / scalar);
    // Special case for division by -1 to handle overflow
    if (std::is_signed<T>::value && scalar == static_cast<T>(-1)) {
        auto vec_min_val = hn::Set(tag, std::numeric_limits<T>::min());
        bool raise_overflow = false;

        for (; len >= lanes; len -= lanes, src += lanes, dst += lanes) {
            auto vec_src = hn::Load(tag, src);
            auto is_min_val = hn::Eq(vec_src, vec_min_val);

            // Apply negation, taking care of overflow
            auto vec_res = hn::IfThenElse(is_min_val, vec_min_val, hn::Sub(hn::Zero(tag), vec_src));
            hn::Store(vec_res, tag, dst);

            if (hn::AllTrue(tag, is_min_val)) {
                raise_overflow = true;
            }
        }

        // Scalar fallback for remaining elements
        for (; len > 0; --len, ++src, ++dst) {
            if (*src == std::numeric_limits<T>::min()) {
                *dst = std::numeric_limits<T>::min();
                raise_overflow = true;
            } else {
                *dst = -*src;
            }
        }

        if (raise_overflow) {
            npy_set_floatstatus_overflow();
        }
        return;
    } else {
        // General case for scalar division
        
        for (; len >= lanes; len -= lanes, src += lanes, dst += lanes) {
            auto vec_src = hn::Load(tag, src);
            auto vec_scalar = hn::Set(tag, scalar);
            auto vec_res = hn::Div(vec_src, vec_scalar);

            // Floor adjust for signed division
            auto vec_mul = hn::Mul(vec_res, vec_scalar);
            auto remainder_check = hn::Ne(vec_src, vec_mul);

            auto vec_nsign_src = hn::Lt(vec_src, hn::Zero(tag));
            auto vec_nsign_scalar = hn::Lt(vec_scalar, hn::Zero(tag));
            auto diff_sign = hn::Xor(vec_nsign_src, vec_nsign_scalar);

            vec_res = hn::IfThenElse(hn::And(remainder_check, diff_sign), hn::Sub(vec_res, hn::Set(tag, 1)), vec_res);
            hn::Store(vec_res, tag, dst);
        }

        // Scalar fallback for remaining elements
        for (; len > 0; --len, ++src, ++dst) {
            const T a = *src;
            T result = a / scalar;

            // Floor adjust for signed division in scalar fallback
            if (std::is_signed<T>::value && ((a > 0) != (scalar > 0)) && (result * scalar != a)) {
                result--;
            }
            *dst = result;
        }
    }
    SIMD_CLEANUP();
}




// SIMD division for unsigned integer types
template <typename T>
void simd_divide_by_scalar_contig_unsigned(T* src, T scalar, T* dst, npy_intp len, hn::ScalableTag<T> tag) {
    const int lanes = hn::Lanes(tag);

    // Precompute the reciprocal (1 / scalar) for optimized division
    auto vec_reciprocal = hn::Set(tag, static_cast<T>(1) / scalar);

    // Vectorized loop for division
    for (; len >= lanes; len -= lanes, src += lanes, dst += lanes) {
        auto vec_src = hn::Load(tag, src);
        // Perform multiplication by reciprocal (optimized division)
        auto vec_res = hn::Mul(vec_src, vec_reciprocal);
        hn::Store(vec_res, tag, dst);
    }
    // Scalar fallback for remaining elements
    for (; len > 0; --len, ++src, ++dst) {
        const T a = *src;
        *dst = a / scalar;  // Fallback to scalar division
    }
    SIMD_CLEANUP();
}


#if NPY_SIMD

// The main wrapper function that checks for architecture-specific conditions
template <typename T>
void simd_divide_by_scalar_contig(T* src, T scalar, T* dst, npy_intp len, hn::ScalableTag<T> tag) {
    // Check if the type is signed or unsigned and call the respective function
    if (std::is_signed<T>::value) {
        simd_divide_by_scalar_contig_signed(src, scalar, dst, len, tag);
    } else {
        simd_divide_by_scalar_contig_unsigned(src, scalar, dst, len, tag);
    }
}
#endif

template <typename T>
void simd_divide(char** args, npy_intp const* dimensions, npy_intp const* steps, hn::ScalableTag<T> tag) {
    // Pre-cast pointers to avoid repeated reinterpret_cast
    T* src1 = reinterpret_cast<T*>(args[0]);
    T* src2 = reinterpret_cast<T*>(args[1]);
    T* dst1 = reinterpret_cast<T*>(args[2]);

    // Check if this is a binary reduction operation
    if (IS_BINARY_REDUCE) {
        BINARY_REDUCE_LOOP(T) {
            if (std::is_signed<T>::value) {
                io1 = floor_div(io1, *src2);
            } else {
                const T d = *src2;
                if (NPY_UNLIKELY(d == 0)) {
                    npy_set_floatstatus_divbyzero();
                    io1 = 0;
                } else {
                    io1 /= d;
                }
            }
            src2 += is2;
        }
        *dst1 = io1;  // Store the result of the reduction
        return;  // Early exit after binary reduction
    }

#if NPY_SIMD
    // SIMD-based division for contiguous memory when divisor is a scalar and not 0
    if (IS_BLOCKABLE_BINARY_SCALAR2(sizeof(T), NPY_SIMD_WIDTH) && (*src2 != 0)) {
        simd_divide_by_scalar_contig(src1, *src2, dst1, dimensions[0], tag);
        return;  // Early exit after SIMD division
    }
#endif

    // General binary loop for non-contiguous or non-scalar cases
    BINARY_LOOP {
        if (std::is_signed<T>::value) {
            *dst1 = floor_div(*src1, *src2);
        } else {
            const T in2 = *src2;
            if (NPY_UNLIKELY(in2 == 0)) {
                npy_set_floatstatus_divbyzero();
                *dst1 = 0;
            } else {
                *dst1 = *src1 / in2;
            }
        }
    }
}


template <typename T>
int simd_divide_indexed(T* ip1, npy_intp* indxp, T* value, npy_intp n, npy_intp is1, npy_intp isindex, npy_intp isb, npy_intp shape, hn::ScalableTag<T> tag) {
    const int lanes = hn::Lanes(tag);

    for (npy_intp i = 0; i < n; i++, indxp += isindex, value += isb) {
        npy_intp indx = *indxp;

        // // Ensure index is within bounds
        // if (indx < 0 || indx >= shape) {
        //     ERROR_INDEX_OUT_OF_BOUNDS();//indx += shape;
        // }

        T* indexed = reinterpret_cast<T*>(ip1 + is1 * indx);

        if (i + lanes <= n) {

            auto vec_indexed = hn::LoadU(tag, indexed);
            auto vec_value = hn::Set(tag, *value);

            if (std::is_signed<T>::value) {
                auto vec_res = hn::Div(vec_indexed, vec_value);
                auto product = hn::Mul(vec_res, vec_value);
                auto remainder_check = hn::Ne(vec_indexed, product);
                auto n_positive = hn::Gt(vec_indexed, hn::Zero(tag));
                auto d_positive = hn::Gt(vec_value, hn::Zero(tag));
                auto diff_sign = hn::Xor(n_positive, d_positive);

                vec_res = hn::IfThenElse(hn::And(remainder_check, diff_sign), hn::Sub(vec_res, hn::Set(tag, 1)), vec_res);
                hn::StoreU(vec_res, tag, indexed);
            } else {
                auto zero_mask = hn::Eq(vec_value, hn::Zero(tag));
                auto vec_res = hn::IfThenElse(zero_mask, hn::Zero(tag), hn::Div(vec_indexed, vec_value));
                hn::StoreU(vec_res, tag, indexed);

                if (hn::AllTrue(tag, zero_mask)) {
                    npy_set_floatstatus_divbyzero();
                }
            }

            i += lanes - 1;
        } else {
            T in2 = *value;
            if (std::is_signed<T>::value) {
                *indexed = floor_div(*indexed, in2);
            } else {
                if (NPY_UNLIKELY(in2 == 0)) {
                    npy_set_floatstatus_divbyzero();
                    *indexed = 0;
                } else {
                    *indexed = *indexed / in2;
                }
            }
        }
    }
    return 0;
}

// Macro to define functions for multiple types
#define DEFINE_SIMD_FUNCTIONS(TYPE, SCALAR_TYPE) \
    extern "C" { \
        NPY_NO_EXPORT void NPY_CPU_DISPATCH_CURFX(TYPE##_divide)(char** args, npy_intp const* dimensions, npy_intp const* steps, void* NPY_UNUSED(func)) { \
            hn::ScalableTag<SCALAR_TYPE> tag; \
            simd_divide<SCALAR_TYPE>(args, dimensions, steps, tag); \
        } \
        NPY_NO_EXPORT int NPY_CPU_DISPATCH_CURFX(TYPE##_divide_indexed)(PyArrayMethod_Context *context, char *const *args, npy_intp const *dimensions, npy_intp const *steps, NpyAuxData *auxdata) { \
            auto ip1 = reinterpret_cast<SCALAR_TYPE*>(args[0]); \
            auto indxp = reinterpret_cast<npy_intp*>(args[1]); \
            auto value = reinterpret_cast<SCALAR_TYPE*>(args[2]); \
            npy_intp is1 = steps[0], isindex = steps[1], isb = steps[2]; \
            npy_intp shape = steps[3]; \
            npy_intp n = dimensions[0]; \
            return simd_divide_indexed(ip1, indxp, value, n, is1, isindex, isb, shape, hn::ScalableTag<SCALAR_TYPE>()); \
        } \
    }

// Define functions for all necessary types
DEFINE_SIMD_FUNCTIONS(BYTE, int8_t)
DEFINE_SIMD_FUNCTIONS(UBYTE, uint8_t)
DEFINE_SIMD_FUNCTIONS(SHORT, int16_t)
DEFINE_SIMD_FUNCTIONS(USHORT, uint16_t)
DEFINE_SIMD_FUNCTIONS(INT, int32_t)
DEFINE_SIMD_FUNCTIONS(UINT, uint32_t)
DEFINE_SIMD_FUNCTIONS(LONG, int64_t)
DEFINE_SIMD_FUNCTIONS(ULONG, uint64_t)
DEFINE_SIMD_FUNCTIONS(LONGLONG, int64_t)
DEFINE_SIMD_FUNCTIONS(ULONGLONG, uint64_t)