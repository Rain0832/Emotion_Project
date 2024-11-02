%clear all;
% 指定Excel文件的路径
% filePath = 'E:\大学项目\lmh预处理\preprocessed_GSR.xlsx';
%filePath = 'E:\2大学项目\所有数据excel\所有原GSR数据（带行列索引）.xlsx';
filePath = 'E:\2大学项目\DEAP\初始可用程序\原始数据对不对\所有原GSR数据检验版（带行列索引）.xlsx';

% 指定工作表名称或索引
sheetNameOrIndex = 'Sheet1'; 
range = '881:901'; %行
% 使用readtable函数读取Excel文件的指定范围
gsr_data = readtable(filePath, 'Sheet', sheetNameOrIndex, 'Range', range);
% pre_GSR_s01 = data{1, 2:8065};
% pre_GSR_s02 = data{2, 2:8065};
for i = 1:20
    % 提取第i行的数据（跳过第一列）
    GSR_s = gsr_data(i, 2:8065); % 使用行和列下标访问表的行和列
    
    % 将表转换为数组
    GSR_s_array = table2array(GSR_s);
    GSR_s_array = GSR_s_array(:);
    
    % 保存为单独的变量
    varName = ['pre_GSR_s', sprintf('%02d', i)];
    % 使用 eval 而不是 assignin
    eval([varName, ' = GSR_s_array;']);
end