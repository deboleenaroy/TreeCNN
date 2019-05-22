function setup_2(varargin)


%run C:\Temp\MATLAB\matconvnet-1.0-beta24\matlab\vl_compilenn; 
run /home/min/a/roy77/DNN/practical-cnn-2016a/vlfeat/toolbox/vl_setup ;
run /home/min/a/roy77/DNN/practical-cnn-2016a/matconvnet/matlab/vl_setupnn ;
addpath /home/min/a/roy77/DNN/practical-cnn-2016a/matconvnet/examples ;


opts.useGpu = false ;
opts.verbose = false ;
opts = vl_argparse(opts, varargin) ;

try
  vl_nnconv(single(1),single(1),[]) ;
catch
  warning('VL_NNCONV() does not seem to be compiled. Trying to compile it now.') ;
  vl_compilenn('enableGpu', opts.useGpu, 'verbose', opts.verbose) ;
end

if opts.useGpu
  try
    vl_nnconv(gpuArray(single(1)),gpuArray(single(1)),[]) ;
  catch
    vl_compilenn('enableGpu', opts.useGpu, 'verbose', opts.verbose) ;
    warning('GPU support does not seem to be compiled in MatConvNet. Trying to compile it now') ;
  end
end
