% 假设images是一个包含图像的cell数组
filelist = dir(['Y:\Training_Data\ESB\', '*.tif']);
% 遍历图像数组计算各通道的均值和方差
num=length(filelist);
% 初始化均值和方差向量
mean_vector = zeros(1,3);
variance_vector = zeros(1,3);
 

parfor i=1:num
    image = imread([filelist(i).folder,'\', filelist(i).name]); % [1.n]
    image = double(image);
    % 计算每个通道的均值
    mean_vector = mean_vector + mean(mean(image));
    % 计算每个通道的方差
    variance_vector = variance_vector + mean(var(image));
end
 
% 将均值和方差分别除以图像数量来获取平均值
mean_vector = mean_vector / length(filelist);
variance_vector = variance_vector / length(filelist);
 
% 打印结果
disp('均值向量:');
disp(mean_vector);
disp('方差向量:');
disp(variance_vector);