function recursive_add_children_v2(node_prev,netPath)

fp = fullfile(netPath,sprintf('%d',node_prev.id));
node_prev.filePath = fp;
filePath = fullfile(fp,'net-best.mat');
ltPath = fullfile(fp,'labelsTransform.mat');
if exist(filePath,'file') & exist(ltPath, 'file')
    node_prev.net = loadnet(filePath);
    node_prev.labelsTransform = loadlt(ltPath);
    for i = 1:numel(unique(node_prev.labelsTransform(:,2)))
        node_prev.children{i} = node;
        node_prev.children{i}.id = node_prev.id*100 + i;
	    node_prev.children{i}.parent = node_prev;
        recursive_add_children_v2(node_prev.children{i}, netPath)
    end

end
