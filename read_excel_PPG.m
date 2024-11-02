% 指定Excel文件的路径
filePath = 'E:\2大学项目\lmh预处理\finally\preprocessed_PPG_final.xlsx';
% 指定工作表名称或索引
sheetNameOrIndex = 'Sheet1'; 
range = '2:41'; %行
% 使用readtable函数读取Excel文件的指定范围
ppg_data = readtable(filePath, 'Sheet', sheetNameOrIndex, 'Range', range);
for i = 1:40
    % 提取第i行的数据（跳过第一列）
    PPG_s = ppg_data(i, 1:8064); %  PPG_s = ppg_data(i, 1:8064)第1列至8064列
    
    % 将表转换为数组
    PPG_s_array = table2array(PPG_s);
    PPG_s_array = PPG_s_array(:);
    
    % 保存为单独的变量
    varName = ['pre_PPG', sprintf('%02d', i)];
    % 使用 eval 而不是 assignin
    eval([varName, ' = PPG_s_array;']);
end