/**
 *
 *  Copyright 2016-2020 Netflix, Inc.
 *
 *     Licensed under the BSD+Patent License (the "License");
 *     you may not use this file except in compliance with the License.
 *     You may obtain a copy of the License at
 *
 *         https://opensource.org/licenses/BSDplusPatent
 *
 *     Unless required by applicable law or agreed to in writing, software
 *     distributed under the License is distributed on an "AS IS" BASIS,
 *     WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *     See the License for the specific language governing permissions and
 *     limitations under the License.
 *
 */

#include <limits.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdbool.h>
#include <math.h>
#include <immintrin.h>

#include "../integer_funque_motion.h"
#include "integer_funque_motion_avx2.h"

/**
 * Note: img1_stride and img2_stride are in terms of (sizeof(double) bytes)
 */
double integer_funque_image_mad_avx2(const dwt2_dtype *img1, const dwt2_dtype *img2, int width, int height, int img1_stride, int img2_stride, float pending_div_factor)
{
    motion_accum_dtype accum = 0;
    int i = 0;
    int j = 0;
    for(i = 0; i < height; ++i)
    {
        motion_interaccum_dtype accum_line = 0;
        for(j = 0; j < width - 16; j =+ 16)
        {
            __m256i img1px = _mm256_loadu_si256((__m256i*) (img1 + i * img1_stride + j));
            __m256i img2px = _mm256_loadu_si256((__m256i*) (img2 + i * img2_stride + j));

            __m256i img1px_lower = _mm256_cvtepu16_epi32(_mm256_castsi256_si128(img1px));
            __m256i img1px_upper = _mm256_cvtepu16_epi32(_mm256_castsi256_si128(_mm256_srli_si256(img1px, 16)));

            __m256i img2px_lower = _mm256_cvtepu16_epi32(_mm256_castsi256_si128(img2px));
            __m256i img2px_upper = _mm256_cvtepu16_epi32(_mm256_castsi256_si128(_mm256_srli_si256(img2px, 16)));

            __m256i img_diff_lower = _mm256_abs_epi32(_mm256_sub_epi32(img1px_lower , img2px_lower));
            __m256i img_diff_upper = _mm256_abs_epi32(_mm256_sub_epi32(img1px_upper , img2px_upper));

            __m256i sum_32x8 = _mm256_add_epi32(img_diff_lower , img_diff_upper);
            __m128i sum128 = _mm_add_epi32(_mm256_castsi256_si128(sum_32x8) , _mm256_extracti128_si256(sum_32x8, 1));
            __m128i sum = _mm_hadd_epi32(sum128, sum128);
            sum = _mm_hadd_epi32(sum, sum);

            accum_line += (motion_interaccum_dtype) _mm_cvtsi128_si32(sum);
            // assuming it is 4k video, max accum_inner is 2^16*3840
        }
        for(; j < width; ++j)
        {
            dwt2_dtype img1px = img1[i * img1_stride + j];
            dwt2_dtype img2px = img2[i * img2_stride + j];

            accum_line += (motion_interaccum_dtype) abs(img1px - img2px);
            // assuming it is 4k video, max accum_inner is 2^16*3840
        }
        accum += (motion_accum_dtype) accum_line;
        // assuming it is 4k video, max accum is 2^16*3840*1920 which uses upto 39bits
    }

    double d_accum = (double) accum / pending_div_factor;
    return (d_accum / (width * height));
}
/**
 * Note: prev_stride and curr_stride are in terms of bytes
 */

int integer_compute_motion_funque_avx2(const dwt2_dtype *prev, const dwt2_dtype *curr, int w, int h, int prev_stride, int curr_stride, int pending_div_factor_arg, double *score)
{

    float pending_div_factor = (1 << pending_div_factor_arg) * 255;

    if (prev_stride % sizeof(dwt2_dtype) != 0)
    {
        printf("error: prev_stride %% sizeof(dwt2_dtype) != 0, prev_stride = %d, sizeof(dwt2_dtype) = %zu.\n", prev_stride, sizeof(dwt2_dtype));
        fflush(stdout);
        goto fail;
    }
    if (curr_stride % sizeof(dwt2_dtype) != 0)
    {
        printf("error: curr_stride %% sizeof(dwt2_dtype) != 0, curr_stride = %d, sizeof(dwt2_dtype) = %zu.\n", curr_stride, sizeof(dwt2_dtype));
        fflush(stdout);
        goto fail;
    }
    // stride for integer_funque_image_mad_c is in terms of (sizeof(dwt2_dtype) bytes)

    *score = integer_funque_image_mad_avx2(prev, curr, w, h, prev_stride / sizeof(dwt2_dtype), curr_stride / sizeof(dwt2_dtype), pending_div_factor);

    return 0;

fail:
    return 1;
}

int integer_compute_mad_funque_avx2(const dwt2_dtype *ref, const dwt2_dtype *dis, int w, int h, int ref_stride, int dis_stride, int pending_div_factor_arg, double *score)
{

    float pending_div_factor = (1 << pending_div_factor_arg) * 255;

    if (ref_stride % sizeof(dwt2_dtype) != 0)
    {
        printf("error: ref_stride %% sizeof(dwt2_dtype) != 0, ref_stride = %d, sizeof(dwt2_dtype) = %zu.\n", ref_stride, sizeof(dwt2_dtype));
        fflush(stdout);
        goto fail;
    }
    if (dis_stride % sizeof(dwt2_dtype) != 0)
    {
        printf("error: dis_stride %% sizeof(dwt2_dtype) != 0, dis_stride = %d, sizeof(dwt2_dtype) = %zu.\n", dis_stride, sizeof(dwt2_dtype));
        fflush(stdout);
        goto fail;
    }
    // stride for integer_funque_image_mad_c is in terms of (sizeof(dwt2_dtype) bytes)

    *score = integer_funque_image_mad_avx2(ref, dis, w, h, ref_stride / sizeof(dwt2_dtype), dis_stride / sizeof(dwt2_dtype), pending_div_factor);

    return 0;

fail:
    return 1;
}
