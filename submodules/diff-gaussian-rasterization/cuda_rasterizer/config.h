/*
 * Copyright (C) 2023, Inria
 * GRAPHDECO research group, https://team.inria.fr/graphdeco
 * All rights reserved.
 *
 * This software is free for non-commercial, research and evaluation use 
 * under the terms of the LICENSE.md file.
 *
 * For inquiries contact  george.drettakis@inria.fr
 */

#ifndef CUDA_RASTERIZER_CONFIG_H_INCLUDED
#define CUDA_RASTERIZER_CONFIG_H_INCLUDED

#define NUM_CHANNELS 3 // Default 3, RGB
#define BLOCK_X 16
#define BLOCK_Y 16

#define CFG_Ld
#define CFG_Wd 100.0f
#define CFG_far 1000.0f
#define CFG_near 0.2f

#define CFG_Ln
#define CFG_Wn 0.05f

// define an macro that represent Ld || Ln, used to compute shared values in backwards
#ifdef CFG_Ld
#define CFG_LdOrLn
#else
#ifdef CFG_Ln
#define CFG_LdOrLn
#endif
#endif

#endif