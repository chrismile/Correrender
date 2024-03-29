/*
 * BSD 2-Clause License
 *
 * Copyright (c) 2023, Christoph Neuhauser
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * * Redistributions of source code must retain the above copyright notice, this
 *   list of conditions and the following disclaimer.
 *
 * * Redistributions in binary form must reproduce the above copyright notice,
 *   this list of conditions and the following disclaimer in the documentation
 *   and/or other materials provided with the distribution.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
 * FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
 * SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 * CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
 * OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

#ifndef CORRERENDER_REPLICABILITYSTATE_HPP
#define CORRERENDER_REPLICABILITYSTATE_HPP

/*
 * State string for visualization of the artificial data set linear_4x4 for the Replicability Stamp.
 * Two different strings are provided, as the 4K window layout doesn't work optimally at other resolutions.
 */
const char REPLICABILITY_STATE_STRING_4K[] =
        "{\n"
        "    \"calculators\" :\n"
        "    [\n"
        "        {\n"
        "            \"state\" :\n"
        "            {\n"
        "                \"calculate_absolute_value\" : \"0\",\n"
        "                \"correlation_measure_type\" : \"pearson\",\n"
        "                \"correlation_mode\" : \"Ensemble\",\n"
        "                \"data_mode\" : \"Buffer Array\",\n"
        "                \"device\" : \"CUDA\",\n"
        "                \"fix_picking_z\" : \"1\",\n"
        "                \"kmi_neighbors\" : \"30\",\n"
        "                \"kraskov_estimator_index\" : \"1\",\n"
        "                \"mi_bins\" : \"80\",\n"
        "                \"reference_point_x\" : \"16\",\n"
        "                \"reference_point_y\" : \"16\",\n"
        "                \"reference_point_z\" : \"16\",\n"
        "                \"scalar_field_idx\" : \"0\",\n"
        "                \"use_buffer_tiling\" : \"1\",\n"
        "                \"use_separate_fields\" : \"0\"\n"
        "            },\n"
        "            \"type\" : \"correlation\"\n"
        "        }\n"
        "    ],\n"
        "    \"dock_data\" : \"[Window][###data_view_0]\\nPos=987,40\\nSize=1408,921\\nCollapsed=0\\nDockId=0x00000005,0\\n\\n[Window][Property Editor]\\nPos=0,40\\nSize=985,1210\\nCollapsed=0\\nDockId=0x00000003,0\\n\\n[Window][Transfer Function]\\nCollapsed=0\\nDockId=0x00000004\\n\\n[Window][Multi-Var Transfer Function]\\nPos=0,1252\\nSize=985,823\\nCollapsed=0\\nDockId=0x00000004,1\\n\\n[Window][Camera Checkpoints]\\nPos=0,1252\\nSize=985,823\\nCollapsed=0\\nDockId=0x00000004,0\\n\\n[Window][Replay Widget]\\nCollapsed=0\\nDockId=0x00000004\\n\\n[Window][DockSpaceViewport_11111111]\\nPos=0,40\\nSize=3840,2035\\nCollapsed=0\\n\\n[Window][Debug##Default]\\nPos=60,60\\nSize=400,400\\nCollapsed=0\\n\\n[Window][Choose a File##ChooseDataSetFile]\\nPos=1309,700\\nSize=1222,674\\nCollapsed=0\\n\\n[Window][Export Field]\\nPos=1777,982\\nSize=648,157\\nCollapsed=0\\n\\n[Window][Choose a File##ChooseExportFieldFile]\\nPos=1309,747\\nSize=1222,580\\nCollapsed=0\\n\\n[Window][Choose TorchScript Model File##ChoosePyTorchModelFile]\\nPos=1357,700\\nSize=1126,674\\nCollapsed=0\\n\\n[Window][Choose QuickMLP Model File##ChooseQuickMLPModelFile]\\nPos=1346,700\\nSize=1148,674\\nCollapsed=0\\n\\n[Window][Choose tiny-cuda-nn Model File##ChooseTinyCudaNNModelFile]\\nPos=1331,700\\nSize=1178,674\\nCollapsed=0\\n\\n[Window][###data_view_1]\\nPos=987,963\\nSize=1408,1112\\nCollapsed=0\\nDockId=0x00000006,0\\n\\n[Window][###data_view_2]\\nPos=2397,963\\nSize=1443,1112\\nCollapsed=0\\nDockId=0x0000000A,0\\n\\n[Window][Optimization Progress]\\nPos=1658,964\\nSize=524,147\\nCollapsed=0\\n\\n[Window][Optimize Transfer Function]\\nPos=2640,973\\nSize=722,1003\\nCollapsed=0\\n\\n[Window][Compute Field Similarity]\\nPos=1542,888\\nSize=756,298\\nCollapsed=0\\n\\n[Window][Choose a File##ChooseStateFile]\\nPos=1369,700\\nSize=1102,674\\nCollapsed=0\\n\\n[Window][The file already exists!##Choose a FileChooseStateFileOverWriteDialog]\\nPos=1729,983\\nSize=381,147\\nCollapsed=0\\n\\n[Window][###data_view_3]\\nPos=2397,40\\nSize=1443,921\\nCollapsed=0\\nDockId=0x00000009,0\\n\\n[Table][0x63A27651,2]\\nRefScale=30\\nColumn 0  Weight=1.0000\\nColumn 1  Width=280\\n\\n[Docking][Data]\\nDockSpace       ID=0x8B93E3BD Window=0xA787BDB4 Pos=0,40 Size=3840,2035 Split=X\\n  DockNode      ID=0x00000007 Parent=0x8B93E3BD SizeRef=2395,2035 Split=X\\n    DockNode    ID=0x00000001 Parent=0x00000007 SizeRef=985,2075 Split=Y\\n      DockNode  ID=0x00000003 Parent=0x00000001 SizeRef=1113,1210 Selected=0x61D81DE4\\n      DockNode  ID=0x00000004 Parent=0x00000001 SizeRef=1113,823 Selected=0xADFDA172\\n    DockNode    ID=0x00000002 Parent=0x00000007 SizeRef=1408,2075 Split=Y Selected=0x2651051C\\n      DockNode  ID=0x00000005 Parent=0x00000002 SizeRef=3043,921 CentralNode=1 Selected=0x2651051C\\n      DockNode  ID=0x00000006 Parent=0x00000002 SizeRef=3043,1112 Selected=0x1B312CAC\\n  DockNode      ID=0x00000008 Parent=0x8B93E3BD SizeRef=1443,2035 Split=Y Selected=0x5C91567C\\n    DockNode    ID=0x00000009 Parent=0x00000008 SizeRef=1741,921 Selected=0x61F17FCC\\n    DockNode    ID=0x0000000A Parent=0x00000008 SizeRef=1741,1112 Selected=0x5C91567C\\n\\n\",\n"
        "    \"global_camera\" :\n"
        "    {\n"
        "        \"fovy\" : 0.92729520797729492,\n"
        "        \"lookat\" :\n"
        "        {\n"
        "            \"x\" : 0,\n"
        "            \"y\" : 0,\n"
        "            \"z\" : 0\n"
        "        },\n"
        "        \"pitch\" : 0,\n"
        "        \"position\" :\n"
        "        {\n"
        "            \"x\" : 0,\n"
        "            \"y\" : 0,\n"
        "            \"z\" : 0.57277601957321167\n"
        "        },\n"
        "        \"yaw\" : -1.5707963705062866\n"
        "    },\n"
        "    \"renderers\" :\n"
        "    [\n"
        "        {\n"
        "            \"state\" :\n"
        "            {\n"
        "                \"line_width\" : \"0.001\",\n"
        "                \"use_depth_cues\" : \"1\",\n"
        "                \"view_visibility\" : \"1000\"\n"
        "            },\n"
        "            \"type\" : \"domain_outline\"\n"
        "        },\n"
        "        {\n"
        "            \"state\" :\n"
        "            {\n"
        "                \"attenuation_coefficient\" : \"100\",\n"
        "                \"nan_handling\" : \"ignore\",\n"
        "                \"selected_field_idx\" : \"1\",\n"
        "                \"step_size\" : \"0.1\",\n"
        "                \"view_visibility\" : \"1000\"\n"
        "            },\n"
        "            \"type\" : \"dvr\"\n"
        "        },\n"
        "        {\n"
        "            \"state\" :\n"
        "            {\n"
        "                \"align_with_parent_window\" : \"1\",\n"
        "                \"beta\" : \"0.75\",\n"
        "                \"cell_distance_range_lower\" : \"0\",\n"
        "                \"cell_distance_range_upper\" : \"5\",\n"
        "                \"color_map_0\" : \"Black-Yellow\",\n"
        "                \"color_map_variance\" : \"Gray\",\n"
        "                \"context_diagram_view\" : \"1\",\n"
        "                \"correlation_measure_type\" : \"pearson\",\n"
        "                \"correlation_mode\" : \"Ensemble\",\n"
        "                \"correlation_range_lower\" : \"0\",\n"
        "                \"correlation_range_upper\" : \"1\",\n"
        "                \"curve_opacity_context\" : \"1\",\n"
        "                \"curve_opacity_focus\" : \"1\",\n"
        "                \"curve_thickness\" : \"1.5\",\n"
        "                \"data_mode\" : \"Buffer Array\",\n"
        "                \"desaturate_unselected_ring\" : \"1\",\n"
        "                \"diagram_radius\" : \"160\",\n"
        "                \"diagram_type\" : \"chords\",\n"
        "                \"downscaling_factor__z\" : \"32\",\n"
        "                \"downscaling_factor_focus_x\" : \"2\",\n"
        "                \"downscaling_factor_focus_y\" : \"2\",\n"
        "                \"downscaling_factor_focus_z\" : \"2\",\n"
        "                \"downscaling_factor_x\" : \"32\",\n"
        "                \"downscaling_factor_y\" : \"32\",\n"
        "                \"downscaling_power_of_two\" : \"1\",\n"
        "                \"focus_diagram_view\" : \"2\",\n"
        "                \"kmi_neighbors\" : \"30\",\n"
        "                \"line_count_factor_context\" : \"1000\",\n"
        "                \"line_count_factor_focus\" : \"1000\",\n"
        "                \"mi_bins\" : \"80\",\n"
        "                \"num_bo_iterations\" : \"100\",\n"
        "                \"num_init_samples\" : \"20\",\n"
        "                \"num_samples\" : \"200\",\n"
        "                \"octree_method\" : \"Top Down (PoT)\",\n"
        "                \"opacity_by_value\" : \"0\",\n"
        "                \"outer_ring_size_pct\" : \"0.1\",\n"
        "                \"render_only_last_focus_diagram\" : \"1\",\n"
        "                \"sampling_method_type\" : \"Bayesian Optimization\",\n"
        "                \"scalar_field_selection\" : \"100\",\n"
        "                \"separate_color_variance_and_correlation\" : \"1\",\n"
        "                \"show_only_selected_variable_in_focus_diagrams\" : \"1\",\n"
        "                \"show_selected_regions_by_color\" : \"1\",\n"
        "                \"use_absolute_correlation_measure\" : \"1\",\n"
        "                \"use_alignment_rotation\" : \"0\",\n"
        "                \"use_buffer_tiling\" : \"1\",\n"
        "                \"use_correlation_computation_gpu\" : \"1\",\n"
        "                \"use_global_std_dev_range\" : \"1\",\n"
        "                \"use_neon_selection_colors\" : \"1\",\n"
        "                \"use_opaque_selection_boxes\" : \"1\",\n"
        "                \"view_visibility\" : \"1110\"\n"
        "            },\n"
        "            \"type\" : \"diagram\"\n"
        "        }\n"
        "    ],\n"
        "    \"views\" :\n"
        "    [\n"
        "        {\n"
        "            \"name\" : \"3D View\",\n"
        "            \"sync_with_global_camera\" : true\n"
        "        },\n"
        "        {\n"
        "            \"name\" : \"Context View\",\n"
        "            \"sync_with_global_camera\" : true\n"
        "        },\n"
        "        {\n"
        "            \"name\" : \"Focus View\",\n"
        "            \"sync_with_global_camera\" : true\n"
        "        },\n"
        "        {\n"
        "            \"name\" : \"Data View\",\n"
        "            \"sync_with_global_camera\" : true\n"
        "        }\n"
        "    ],\n"
        "    \"volume_data\" :\n"
        "    {\n"
        "        \"current_ensemble_idx\" : 0,\n"
        "        \"current_time_step_idx\" : 0,\n"
        "        \"name\" : \"linear_4x4\",\n"
        "        \"transfer_functions\" :\n"
        "        [\n"
        "            {\n"
        "                \"data\" : \"<TransferFunction colorspace=\\\"sRGB\\\" interpolation_colorspace=\\\"Linear RGB\\\">\\n    <OpacityPoints>\\n        <OpacityPoint position=\\\"0\\\" opacity=\\\"1\\\"/>\\n        <OpacityPoint position=\\\"1\\\" opacity=\\\"1\\\"/>\\n    </OpacityPoints>\\n    <ColorPoints color_data=\\\"ushort\\\">\\n        <ColorPoint position=\\\"0\\\" r=\\\"15163\\\" g=\\\"19532\\\" b=\\\"49344\\\"/>\\n        <ColorPoint position=\\\"0.25\\\" r=\\\"37008\\\" g=\\\"45746\\\" b=\\\"65278\\\"/>\\n        <ColorPoint position=\\\"0.5\\\" r=\\\"56540\\\" g=\\\"56540\\\" b=\\\"56540\\\"/>\\n        <ColorPoint position=\\\"0.75\\\" r=\\\"62965\\\" g=\\\"40092\\\" b=\\\"32125\\\"/>\\n        <ColorPoint position=\\\"1\\\" r=\\\"46260\\\" g=\\\"1028\\\" b=\\\"9766\\\"/>\\n    </ColorPoints>\\n</TransferFunction>\\n\\u0000\",\n"
        "                \"is_selected_range_fixed\" : false,\n"
        "                \"selected_range\" :\n"
        "                {\n"
        "                    \"max\" : 4.4554667472839355,\n"
        "                    \"min\" : -4.880955696105957\n"
        "                }\n"
        "            },\n"
        "            {\n"
        "                \"data\" : \"<TransferFunction colorspace=\\\"sRGB\\\" interpolation_colorspace=\\\"Linear RGB\\\">\\n    <OpacityPoints>\\n        <OpacityPoint position=\\\"0\\\" opacity=\\\"1\\\"/>\\n        <OpacityPoint position=\\\"0.49783080816268921\\\" opacity=\\\"0\\\"/>\\n        <OpacityPoint position=\\\"1\\\" opacity=\\\"1\\\"/>\\n    </OpacityPoints>\\n    <ColorPoints color_data=\\\"ushort\\\">\\n        <ColorPoint position=\\\"0\\\" r=\\\"15163\\\" g=\\\"19532\\\" b=\\\"49344\\\"/>\\n        <ColorPoint position=\\\"0.25\\\" r=\\\"37008\\\" g=\\\"45746\\\" b=\\\"65278\\\"/>\\n        <ColorPoint position=\\\"0.5\\\" r=\\\"56540\\\" g=\\\"56540\\\" b=\\\"56540\\\"/>\\n        <ColorPoint position=\\\"0.75\\\" r=\\\"62965\\\" g=\\\"40092\\\" b=\\\"32125\\\"/>\\n        <ColorPoint position=\\\"1\\\" r=\\\"46260\\\" g=\\\"1028\\\" b=\\\"9766\\\"/>\\n    </ColorPoints>\\n</TransferFunction>\\n\\u0000\",\n"
        "                \"is_selected_range_fixed\" : false,\n"
        "                \"selected_range\" :\n"
        "                {\n"
        "                    \"max\" : 1,\n"
        "                    \"min\" : -1\n"
        "                }\n"
        "            }\n"
        "        ]\n"
        "    }\n"
        "}\n"
;

const char REPLICABILITY_STATE_STRING_MISC[] =
        "{\n"
        "    \"calculators\" :\n"
        "    [\n"
        "        {\n"
        "            \"state\" :\n"
        "            {\n"
        "                \"calculate_absolute_value\" : \"0\",\n"
        "                \"correlation_measure_type\" : \"pearson\",\n"
        "                \"correlation_mode\" : \"Ensemble\",\n"
        "                \"data_mode\" : \"Buffer Array\",\n"
        "                \"device\" : \"CUDA\",\n"
        "                \"fix_picking_z\" : \"1\",\n"
        "                \"kmi_neighbors\" : \"30\",\n"
        "                \"kraskov_estimator_index\" : \"1\",\n"
        "                \"mi_bins\" : \"80\",\n"
        "                \"reference_point_x\" : \"16\",\n"
        "                \"reference_point_y\" : \"16\",\n"
        "                \"reference_point_z\" : \"16\",\n"
        "                \"scalar_field_idx\" : \"0\",\n"
        "                \"use_buffer_tiling\" : \"1\",\n"
        "                \"use_separate_fields\" : \"0\"\n"
        "            },\n"
        "            \"type\" : \"correlation\"\n"
        "        }\n"
        "    ],\n"
        "    \"dock_data\" : \"[Window][###data_view_0]\\nPos=837,40\\nSize=3003,2035\\nCollapsed=0\\nDockId=0x00000005,0\\n\\n[Window][Property Editor]\\nPos=0,40\\nSize=835,1210\\nCollapsed=0\\nDockId=0x00000003,0\\n\\n[Window][Transfer Function]\\nCollapsed=0\\nDockId=0x00000004\\n\\n[Window][Multi-Var Transfer Function]\\nPos=0,1252\\nSize=835,823\\nCollapsed=0\\nDockId=0x00000004,1\\n\\n[Window][Camera Checkpoints]\\nPos=0,1252\\nSize=835,823\\nCollapsed=0\\nDockId=0x00000004,0\\n\\n[Window][Replay Widget]\\nCollapsed=0\\nDockId=0x00000004\\n\\n[Window][DockSpaceViewport_11111111]\\nPos=0,40\\nSize=3840,2035\\nCollapsed=0\\n\\n[Window][Debug##Default]\\nPos=60,60\\nSize=400,400\\nCollapsed=0\\n\\n[Window][Choose a File##ChooseDataSetFile]\\nPos=1309,700\\nSize=1222,674\\nCollapsed=0\\n\\n[Window][Export Field]\\nPos=1777,982\\nSize=648,157\\nCollapsed=0\\n\\n[Window][Choose a File##ChooseExportFieldFile]\\nPos=1309,747\\nSize=1222,580\\nCollapsed=0\\n\\n[Window][Choose TorchScript Model File##ChoosePyTorchModelFile]\\nPos=1357,700\\nSize=1126,674\\nCollapsed=0\\n\\n[Window][Choose QuickMLP Model File##ChooseQuickMLPModelFile]\\nPos=1346,700\\nSize=1148,674\\nCollapsed=0\\n\\n[Window][Choose tiny-cuda-nn Model File##ChooseTinyCudaNNModelFile]\\nPos=1331,700\\nSize=1178,674\\nCollapsed=0\\n\\n[Window][###data_view_1]\\nPos=2382,40\\nSize=1458,2035\\nCollapsed=0\\nDockId=0x00000006,0\\n\\n[Window][###data_view_2]\\nPos=2397,40\\nSize=1443,2035\\nCollapsed=0\\nDockId=0x0000000A,0\\n\\n[Window][Optimization Progress]\\nPos=1658,964\\nSize=524,147\\nCollapsed=0\\n\\n[Window][Optimize Transfer Function]\\nPos=2640,973\\nSize=722,1003\\nCollapsed=0\\n\\n[Window][Choose a File##ChooseStateFile]\\nPos=1369,700\\nSize=1102,674\\nCollapsed=0\\n\\n[Window][The file already exists!##Choose a FileChooseStateFileOverWriteDialog]\\nPos=1729,983\\nSize=381,147\\nCollapsed=0\\n\\n[Window][Compute Field Similarity]\\nPos=1542,888\\nSize=756,298\\nCollapsed=0\\n\\n[Window][###data_view_3]\\nPos=2397,40\\nSize=1443,921\\nCollapsed=0\\nDockId=0x00000009,0\\n\\n[Window][###data_view_4]\\nPos=3270,1119\\nSize=570,956\\nCollapsed=0\\n\\n[Table][0x63A27651,2]\\nRefScale=30\\nColumn 0  Weight=1.0000\\nColumn 1  Width=280\\n\\n[Docking][Data]\\nDockSpace       ID=0x8B93E3BD Window=0xA787BDB4 Pos=0,40 Size=3840,2035 Split=X\\n  DockNode      ID=0x00000007 Parent=0x8B93E3BD SizeRef=2395,2035 Split=X\\n    DockNode    ID=0x00000001 Parent=0x00000007 SizeRef=835,2075 Split=Y\\n      DockNode  ID=0x00000003 Parent=0x00000001 SizeRef=1113,1210 Selected=0x61D81DE4\\n      DockNode  ID=0x00000004 Parent=0x00000001 SizeRef=1113,823 Selected=0xADFDA172\\n    DockNode    ID=0x00000002 Parent=0x00000007 SizeRef=3003,2075 Split=X Selected=0x2651051C\\n      DockNode  ID=0x00000005 Parent=0x00000002 SizeRef=1543,2035 CentralNode=1 Selected=0x2651051C\\n      DockNode  ID=0x00000006 Parent=0x00000002 SizeRef=1458,2035 Selected=0xC3FCE804\\n  DockNode      ID=0x00000008 Parent=0x8B93E3BD SizeRef=1443,2035 Split=Y Selected=0x5C91567C\\n    DockNode    ID=0x00000009 Parent=0x00000008 SizeRef=1741,921 Selected=0x2DF28928\\n    DockNode    ID=0x0000000A Parent=0x00000008 SizeRef=1741,1112 Selected=0x5AF5B9BE\\n\\n\",\n"
        "    \"global_camera\" :\n"
        "    {\n"
        "        \"fovy\" : 0.92729520797729492,\n"
        "        \"lookat\" :\n"
        "        {\n"
        "            \"x\" : 0.0,\n"
        "            \"y\" : 0.0,\n"
        "            \"z\" : 0.0\n"
        "        },\n"
        "        \"pitch\" : 0.0,\n"
        "        \"position\" :\n"
        "        {\n"
        "            \"x\" : 0.0,\n"
        "            \"y\" : 0.0,\n"
        "            \"z\" : 0.80000001192092896\n"
        "        },\n"
        "        \"yaw\" : -1.5707963705062866\n"
        "    },\n"
        "    \"renderers\" :\n"
        "    [\n"
        "        {\n"
        "            \"state\" :\n"
        "            {\n"
        "                \"line_width\" : \"0.001\",\n"
        "                \"use_depth_cues\" : \"1\",\n"
        "                \"view_visibility\" : \"1\"\n"
        "            },\n"
        "            \"type\" : \"domain_outline\"\n"
        "        },\n"
        "        {\n"
        "            \"state\" :\n"
        "            {\n"
        "                \"attenuation_coefficient\" : \"100\",\n"
        "                \"nan_handling\" : \"ignore\",\n"
        "                \"selected_field_idx\" : \"1\",\n"
        "                \"step_size\" : \"0.1\",\n"
        "                \"view_visibility\" : \"1\"\n"
        "            },\n"
        "            \"type\" : \"dvr\"\n"
        "        },\n"
        "        {\n"
        "            \"state\" :\n"
        "            {\n"
        "                \"align_with_parent_window\" : \"0\",\n"
        "                \"beta\" : \"0.75\",\n"
        "                \"cell_distance_range_lower\" : \"0\",\n"
        "                \"cell_distance_range_upper\" : \"5\",\n"
        "                \"color_map_0\" : \"Black-Yellow\",\n"
        "                \"color_map_variance\" : \"Gray\",\n"
        "                \"context_diagram_view\" : \"0\",\n"
        "                \"correlation_measure_type\" : \"pearson\",\n"
        "                \"correlation_mode\" : \"Ensemble\",\n"
        "                \"correlation_range_lower\" : \"0\",\n"
        "                \"correlation_range_upper\" : \"1\",\n"
        "                \"curve_opacity_context\" : \"1\",\n"
        "                \"curve_opacity_focus\" : \"1\",\n"
        "                \"curve_thickness\" : \"1.5\",\n"
        "                \"data_mode\" : \"Buffer Array\",\n"
        "                \"desaturate_unselected_ring\" : \"1\",\n"
        "                \"diagram_radius\" : \"160\",\n"
        "                \"diagram_type\" : \"chords\",\n"
        "                \"downscaling_factor__z\" : \"32\",\n"
        "                \"downscaling_factor_focus_x\" : \"2\",\n"
        "                \"downscaling_factor_focus_y\" : \"2\",\n"
        "                \"downscaling_factor_focus_z\" : \"2\",\n"
        "                \"downscaling_factor_x\" : \"32\",\n"
        "                \"downscaling_factor_y\" : \"32\",\n"
        "                \"downscaling_power_of_two\" : \"1\",\n"
        "                \"focus_diagram_view\" : \"0\",\n"
        "                \"kmi_neighbors\" : \"30\",\n"
        "                \"line_count_factor_context\" : \"1000\",\n"
        "                \"line_count_factor_focus\" : \"1000\",\n"
        "                \"mi_bins\" : \"80\",\n"
        "                \"num_bo_iterations\" : \"100\",\n"
        "                \"num_init_samples\" : \"20\",\n"
        "                \"num_samples\" : \"200\",\n"
        "                \"octree_method\" : \"Top Down (PoT)\",\n"
        "                \"opacity_by_value\" : \"0\",\n"
        "                \"outer_ring_size_pct\" : \"0.1\",\n"
        "                \"render_only_last_focus_diagram\" : \"1\",\n"
        "                \"sampling_method_type\" : \"Bayesian Optimization\",\n"
        "                \"scalar_field_selection\" : \"100\",\n"
        "                \"separate_color_variance_and_correlation\" : \"1\",\n"
        "                \"show_only_selected_variable_in_focus_diagrams\" : \"1\",\n"
        "                \"show_selected_regions_by_color\" : \"1\",\n"
        "                \"use_absolute_correlation_measure\" : \"1\",\n"
        "                \"use_alignment_rotation\" : \"0\",\n"
        "                \"use_buffer_tiling\" : \"1\",\n"
        "                \"use_correlation_computation_gpu\" : \"1\",\n"
        "                \"use_global_std_dev_range\" : \"1\",\n"
        "                \"use_neon_selection_colors\" : \"1\",\n"
        "                \"use_opaque_selection_boxes\" : \"1\",\n"
        "                \"use_separate_sampling_method_focus\" : \"0\",\n"
        "                \"view_visibility\" : \"1\"\n"
        "            },\n"
        "            \"type\" : \"diagram\"\n"
        "        }\n"
        "    ],\n"
        "    \"views\" :\n"
        "    [\n"
        "        {\n"
        "            \"name\" : \"3D View\",\n"
        "            \"sync_with_global_camera\" : true\n"
        "        }\n"
        "    ],\n"
        "    \"volume_data\" :\n"
        "    {\n"
        "        \"current_ensemble_idx\" : 0,\n"
        "        \"current_time_step_idx\" : 0,\n"
        "        \"name\" : \"linear_4x4\",\n"
        "        \"transfer_functions\" :\n"
        "        [\n"
        "            {\n"
        "                \"data\" : \"<TransferFunction colorspace=\\\"sRGB\\\" interpolation_colorspace=\\\"Linear RGB\\\">\\n    <OpacityPoints>\\n        <OpacityPoint position=\\\"0\\\" opacity=\\\"1\\\"/>\\n        <OpacityPoint position=\\\"1\\\" opacity=\\\"1\\\"/>\\n    </OpacityPoints>\\n    <ColorPoints color_data=\\\"ushort\\\">\\n        <ColorPoint position=\\\"0\\\" r=\\\"15163\\\" g=\\\"19532\\\" b=\\\"49344\\\"/>\\n        <ColorPoint position=\\\"0.25\\\" r=\\\"37008\\\" g=\\\"45746\\\" b=\\\"65278\\\"/>\\n        <ColorPoint position=\\\"0.5\\\" r=\\\"56540\\\" g=\\\"56540\\\" b=\\\"56540\\\"/>\\n        <ColorPoint position=\\\"0.75\\\" r=\\\"62965\\\" g=\\\"40092\\\" b=\\\"32125\\\"/>\\n        <ColorPoint position=\\\"1\\\" r=\\\"46260\\\" g=\\\"1028\\\" b=\\\"9766\\\"/>\\n    </ColorPoints>\\n</TransferFunction>\\n\\u0000\",\n"
        "                \"is_selected_range_fixed\" : false,\n"
        "                \"selected_range\" :\n"
        "                {\n"
        "                    \"max\" : 4.4554667472839355,\n"
        "                    \"min\" : -4.880955696105957\n"
        "                }\n"
        "            },\n"
        "            {\n"
        "                \"data\" : \"<TransferFunction colorspace=\\\"sRGB\\\" interpolation_colorspace=\\\"Linear RGB\\\">\\n    <OpacityPoints>\\n        <OpacityPoint position=\\\"0\\\" opacity=\\\"1\\\"/>\\n        <OpacityPoint position=\\\"0.49783080816268921\\\" opacity=\\\"0\\\"/>\\n        <OpacityPoint position=\\\"1\\\" opacity=\\\"1\\\"/>\\n    </OpacityPoints>\\n    <ColorPoints color_data=\\\"ushort\\\">\\n        <ColorPoint position=\\\"0\\\" r=\\\"15163\\\" g=\\\"19532\\\" b=\\\"49344\\\"/>\\n        <ColorPoint position=\\\"0.25\\\" r=\\\"37008\\\" g=\\\"45746\\\" b=\\\"65278\\\"/>\\n        <ColorPoint position=\\\"0.5\\\" r=\\\"56540\\\" g=\\\"56540\\\" b=\\\"56540\\\"/>\\n        <ColorPoint position=\\\"0.75\\\" r=\\\"62965\\\" g=\\\"40092\\\" b=\\\"32125\\\"/>\\n        <ColorPoint position=\\\"1\\\" r=\\\"46260\\\" g=\\\"1028\\\" b=\\\"9766\\\"/>\\n    </ColorPoints>\\n</TransferFunction>\\n\\u0000\",\n"
        "                \"is_selected_range_fixed\" : false,\n"
        "                \"selected_range\" :\n"
        "                {\n"
        "                    \"max\" : 1.0,\n"
        "                    \"min\" : -1.0\n"
        "                }\n"
        "            }\n"
        "        ]\n"
        "    }\n"
        "}\n"
;

#endif //CORRERENDER_REPLICABILITYSTATE_HPP
