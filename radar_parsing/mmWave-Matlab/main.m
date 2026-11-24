%  Copyright (C) 2018 Texas Instruments Incorporated - http://www.ti.com/
%
%
%   Redistribution and use in source and binary forms, with or without
%   modification, are permitted provided that the following conditions
%   are met:
%
%     Redistributions of source code must retain the above copyright
%     notice, this list of conditions and the following disclaimer.
%
%     Redistributions in binary form must reproduce the above copyright
%     notice, this list of conditions and the following disclaimer in the
%     documentation and/or other materials provided with the
%     distribution.
%
%     Neither the name of Texas Instruments Incorporated nor the names of
%     its contributors may be used to endorse or promote products derived
%     from this software without specific prior written permission.
%
%   THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
%   "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
%   LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
%   A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
%   OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
%   SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
%   LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
%   DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
%   THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
%   (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
%   OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
%


clearvars
close all

%% File paths
sequence_dir = input('Path to the extracted sequence: ', 's');
radar_dir = fullfile(sequence_dir, 'radar');
data_folder = fullfile(radar_dir, 'bin');
output_folder = fullfile(radar_dir, 'mat');
assert(isfolder(data_folder), 'Missing TI capture directory: %s', data_folder);

process_4d = 1;
process_pcl = 0;
if process_4d && ~isfolder(output_folder)
    mkdir(output_folder);
end
if process_pcl && ~isfolder(fullfile(radar_dir, 'radarpcl'))
    mkdir(fullfile(radar_dir, 'radarpcl'));
end

param_path = fullfile('main', 'cascade', 'input', 'hardware_param.m');
calib_path = fullfile('main', 'cascade', 'input', 'calibrateResults_high.mat');
process_cas(data_folder, output_folder, param_path, calib_path, radar_dir, process_4d, process_pcl);
