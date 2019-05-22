function [net,error] = cnn_my_test(net, imdb, opts)

if isempty(opts.test), opts.test = find(imdb.images.set==1) ; end

if ischar(opts.errorFunction)
  switch opts.errorFunction
    case 'none'
      opts.errorFunction = @error_none ;
      hasError = false ;
    case 'multiclass'
      opts.errorFunction = @error_multiclass ;
      if isempty(opts.errorLabels), opts.errorLabels = {'top1err', 'top5err'} ; end
    case 'binary'
      opts.errorFunction = @error_binary ;
      if isempty(opts.errorLabels), opts.errorLabels = {'binerr'} ; end
  end
end
error = struct('top1err', 0, 'top5err', 0, 'labels', [], 'predictions', []);
num = 0;
%net.layers(end) = [];
if numel(opts.gpus) > 1
  parserv = ParameterServer(optstop.parameterServer) ;
  vl_simplenn_start_parserv(net, parserv) ;
else
  parserv = [] ;
end

for t = 1:opts.batchSize:numel(opts.test)
    %batchSize = min(opts.batchSize, numel(opts.test) - t + 1) ;
    batchStart = t;
    batchEnd = min(t+opts.batchSize-1, numel(opts.test)) ;
    batch = opts.test(batchStart : 1 : batchEnd) ;
    num = num + batchEnd - batchStart + 1;
    
    if numel(batch) == 0, continue ; end
    [im, labels] = getBatch(imdb, batch,opts) ;
    net.layers{end}.class = labels ;
    res = [];
    dzdy = [];
    res = vl_simplenn(net, im, dzdy, res, ...
                      'accumulate', false, ...
                      'mode', 'test', ...
                      'conserveMemory', opts.conserveMemory, ...
                      'backPropDepth', opts.backPropDepth, ...
                      'sync', opts.sync, ...
                      'cudnn', opts.cudnn, ...
                      'parameterServer', parserv, ...
                      'holdOn', false);
    err = opts.errorFunction(opts, labels, res);
    error.top1err = error.top1err + err.top1err;
    error.top5err = error.top5err + err.top5err;
    error.labels = cat(2, error.labels, err.labels);
    error.predictions = cat(2, error.predictions, err.predictions);
    
    fprintf('Testing: Batch %d / %d ; top1err : %.3f top5err : %.3f \n', ...
             fix((t-1)/opts.batchSize)+1, ceil(numel(opts.test)/opts.batchSize), ...
             error.top1err/num, error.top5err/num);
end
end


function err = error_multiclass(params, labels, res)
% -------------------------------------------------------------------------
predictions = gather(res(end-1).x) ;
[~,predictions] = sort(predictions, 3, 'descend') ;

% be resilient to badly formatted labels
if numel(labels) == size(predictions, 4)
  labels = reshape(labels,1,1,1,[]) ;
end

% skip null labels
mass = single(labels(:,:,1,:) > 0) ;
if size(labels,3) == 2
  % if there is a second channel in labels, used it as weights
  mass = mass .* labels(:,:,2,:) ;
  labels(:,:,2,:) = [] ;
end

m = min(5, size(predictions,3)) ;

error = ~bsxfun(@eq, predictions, labels) ;
err.top1err = sum(sum(sum(mass .* error(:,:,1,:)))) ;
err.top5err = sum(sum(sum(mass .* min(error(:,:,1:m,:),[],3)))) ;
err.predictions(1,:) = predictions(1,1,1,:);
err.labels(1,:) = labels(1,1,1,:);


end

% -------------------------------------------------------------------------
% function [images, labels] = getBatch(imdb, batch)
% % -------------------------------------------------------------------------
% images = imdb.images.data(:,:,:,batch) ;
% labels = imdb.images.labels(1,batch) ;
% end
