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
#define DEFAULT_MOTION_LEVELS   4
#define DEFAULT_MAD_LEVELS   4

int compute_motion_funque(const float *prev, const float *curr, int w, int h, int ref_stride, int dis_stride, double *score);

int compute_mad_funque(const float *ref, const float *dis, int w, int h, int ref_stride, int dis_stride, double *score);
