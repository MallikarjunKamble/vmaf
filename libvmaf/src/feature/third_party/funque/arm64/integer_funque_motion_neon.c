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

#include <stdlib.h>
#include <arm_neon.h>

#include "integer_funque_motion.h"
#include "../integer_funque_filters.h"
/**
 * Note: img1_stride and img2_stride are in terms of (sizeof(double) bytes)
 */
double integer_funque_image_mad_neon(const dwt2_dtype *img1, const dwt2_dtype *img2, int width, int height, int img1_stride, int img2_stride, float pending_div_factor)
{
    motion_accum_dtype accum = 0;
    int i = 0;
    int j = 0;
    for(i = 0; i < height; ++i)
    {
        motion_interaccum_dtype accum_line = 0;
        for(j = 0; j < width - 8; j += 8)
        {
            int16x8_t img1px_8 = vld1q_s16(&img1[i * img1_stride + j]);
            int16x8_t img2px_8 = vld1q_s16(&img2[i * img2_stride + j]);

            int32x4_t abs_diff_high = vabdl_high_s16(img1px_8, img2px_8);
            int32x4_t abs_diff_low = vabdl_s16(vget_low_s16(img1px_8), vget_low_s16(img2px_8));

            accum_line += (motion_interaccum_dtype) vpaddd_s64(vpaddlq_s32(vaddq_s32(abs_diff_high, abs_diff_low)));
        }
        for(; j < width; ++j)
        {
            dwt2_dtype img1px = img1[i * img1_stride + j];
            dwt2_dtype img2px = img2[i * img2_stride + j];

            accum_line += (motion_interaccum_dtype) abs(img1px - img2px);
        }
        accum += (motion_accum_dtype) accum_line;
    }

    double d_accum = (double) accum / pending_div_factor;
    return (d_accum / (width * height));
}
/**
 * Note: prev_stride and curr_stride are in terms of bytes
 */

int integer_compute_motion_funque_neon(const dwt2_dtype *prev, const dwt2_dtype *curr, int w, int h, int prev_stride, int curr_stride, int pending_div_factor_arg, double *score)
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

    *score = integer_funque_image_mad_neon(prev, curr, w, h, prev_stride / sizeof(dwt2_dtype), curr_stride / sizeof(dwt2_dtype), pending_div_factor);

    return 0;

fail:
    return 1;
}

int integer_compute_mad_funque_neon(const dwt2_dtype *ref, const dwt2_dtype *dis, int w, int h, int ref_stride, int dis_stride, int pending_div_factor_arg, double *score)
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

    *score = integer_funque_image_mad_neon(ref, dis, w, h, ref_stride / sizeof(dwt2_dtype), dis_stride / sizeof(dwt2_dtype), pending_div_factor);

    return 0;

fail:
    return 1;
}
