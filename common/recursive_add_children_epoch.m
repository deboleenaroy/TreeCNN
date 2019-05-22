function recursive_add_children_epoch(node_prev,netPath,epoch)

fp = fullfile(netPath,sprintf('%d',node_prev.id));
node_prev.filePath = fp;
if exist(fullfile(fp,sprintf('net-epoch-%d.mat',epoch)), 'file')
    filePath = fullfile(fp,sprintf('net-epoch-%d.mat',epoch));
else
    filePath = fullfile(fp,'net.mat');
end
ltPath = fullfile(fp,'labelsTransform.mat');
if exist(filePath,'file')
    node_prev.net = loadnet(filePath);
    node_prev.labelsTransform = loadlt(ltPath);
    for i = 1:numel(unique(node_prev.labelsTransform(:,2)))
        node_prev.children{i} = node;
        node_prev.children{i}.id = node_prev.id*100 + i;
        node_prev.children{i}.parent = node_prev;
        recursive_add_children_epoch(node_prev.children{i}, netPath, epoch)
    end

end
