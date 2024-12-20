function varargout = GUI(varargin)

gui_Singleton = 1;
gui_State = struct('gui_Name',       mfilename, ...
                   'gui_Singleton',  gui_Singleton, ...
                   'gui_OpeningFcn', @GUI_OpeningFcn, ...
                   'gui_OutputFcn',  @GUI_OutputFcn, ...
                   'gui_LayoutFcn',  [] , ...
                   'gui_Callback',   []);
if nargin && ischar(varargin{1})
    gui_State.gui_Callback = str2func(varargin{1});
end

if nargout
    [varargout{1:nargout}] = gui_mainfcn(gui_State, varargin{:});
else
    gui_mainfcn(gui_State, varargin{:});
end
% End initialization code - DO NOT EDIT


% --- Executes just before GUI is made visible.
function GUI_OpeningFcn(hObject, eventdata, handles, varargin)

handles.output = hObject;

% Update handles structure
guidata(hObject, handles);


function varargout = GUI_OutputFcn(hObject, eventdata, handles) 

varargout{1} = handles.output;


% --- Executes on button press in pushbutton1.
function pushbutton1_Callback(hObject, eventdata, handles)
    global img;
    axes(handles.axes1);
    cla reset;
    [filename, pathname] = uigetfile('*.jpg;*.png;*.bmp', '选择图片');
    path = [pathname, filename];
    img = imread(path);
    axes(handles.axes1);
    imshow(img);


% --- Executes on button press in pushbutton2.
function pushbutton2_Callback(hObject, eventdata, handles)

    global img2;
    axes(handles.axes2);
    cla reset;
    [filename, pathname] = uigetfile('*.jpg;*.png;*.bmp', '选择图片');
    path = [pathname, filename];
    img2 = imread(path);
    axes(handles.axes2);
    imshow(img2);

    


% --- Executes on button press in pushbutton3.
function pushbutton3_Callback(hObject, eventdata, handles)
    global img;
    if size(img, 3) == 3
        % 获取红色、绿色和蓝色通道
        R = double(img(:,:,1));
        G = double(img(:,:,2));
        B = double(img(:,:,3));    
        % 手动计算灰度图像
        gray_img = 0.299 * R + 0.587 * G + 0.114 * B;
        
        % 将灰度图像转换为uint8类型
        gray_img = uint8(gray_img);
    end
    axes(handles.axes3);
    imshow(gray_img);
    title('灰度化图片');


function pushbutton4_Callback(hObject, eventdata, handles)
     global img; 
    
    % 弹出对话框输入缩放因子
    prompt = {'请输入高度缩放因子（例如：1.5）：', '请输入宽度缩放因子（例如：2）：'};
    dlg_title = '输入缩放因子';
    num_lines = 1;
    defaultans = {'1.5', '1.5'};  % 默认值
    answer = inputdlg(prompt, dlg_title, num_lines, defaultans);
    
    % 缩放因子并转换为数字
    scale_factor_height = str2double(answer{1});
    scale_factor_width = str2double(answer{2});
    
    % 检查输入是否有效
    if isnan(scale_factor_height) || isnan(scale_factor_width)
        errordlg('请输入有效的数字作为缩放因子！', '错误'); % 输入无效则弹出错误对话框
        return;
    end
    
    % 获取原图的尺寸
    [original_height, original_width, num_channels] = size(img);

    % 计算缩放后的尺寸
    new_height = round(original_height * scale_factor_height);
    new_width = round(original_width * scale_factor_width);

    % 创建一个空的图像数组，用于存储缩放后的图像
    img_resized = zeros(new_height, new_width, num_channels, 'uint8');

    % 最近邻插值：遍历每个新图像的像素，找到对应的原图像像素
    for i = 1:new_height
        for j = 1:new_width
            % 计算原图像中的对应位置
            orig_i = round(i / scale_factor_height);
            orig_j = round(j / scale_factor_width);

            % 确保索引在图像的有效范围内
            orig_i = min(max(orig_i, 1), original_height);
            orig_j = min(max(orig_j, 1), original_width);

            % 将原图像的像素值赋值给新图像的像素
            img_resized(i, j, :) = img(orig_i, orig_j, :);
        end
    end

    % 在 axes2 上显示缩放后的图像
    axes(handles.axes3);
    cla reset;
    imshow(img_resized);
    title(['缩放后的图片， 高度因子：', num2str(scale_factor_height), '，宽度因子：', num2str(scale_factor_width)]);


% --- Executes on button press in pushbutton5.
function pushbutton5_Callback(hObject, eventdata, handles)
     global img;
    
    % 输入的旋转角度
    prompt = '请输入旋转角度（度数）：'; % 提示框
    angle = str2double(inputdlg(prompt)); 
    
    if isnan(angle)
        errordlg('请输入有效的数字角度！', '错误'); % 如果输入无效，显示错误信息
        return;
    end
    
    % 将角度转换为弧度
    theta = deg2rad(angle);
    
    % 获取原图的尺寸
    [rows, cols, channels] = size(img);
    
    % 创建一个新的空图像矩阵，用来存放旋转后的图像
    img_rotated = uint8(zeros(rows, cols, channels)); 
    
    % 计算图像中心
    center_x = floor(cols / 2);
    center_y = floor(rows / 2);
    
    % 旋转矩阵（逆旋转）
    rotation_matrix = [cos(theta), -sin(theta); sin(theta), cos(theta)];
    
    % 对每个像素进行旋转操作
    for x = 1:cols
        for y = 1:rows
            % 计算当前像素的相对坐标
            offset_x = x - center_x;
            offset_y = y - center_y;
            
            % 旋转坐标
            new_coords = rotation_matrix * [offset_x; offset_y];
            new_x = round(new_coords(1) + center_x);
            new_y = round(new_coords(2) + center_y);
            
            % 判断新坐标是否在图像内
            if new_x > 0 && new_x <= cols && new_y > 0 && new_y <= rows
                % 将旋转后的位置的像素值复制到新图像中
                img_rotated(y, x, :) = img(new_y, new_x, :);
            end
        end
    end
    
    % 显示旋转后的图像
    axes(handles.axes3);
    cla reset;  % 清空当前坐标轴
    imshow(img_rotated);  % 显示旋转后的图像
    title(['旋转角度：', num2str(angle), '°']);  % 显示旋转角度


% --- Executes on selection change in popupmenu1.
function popupmenu1_Callback(hObject, eventdata, handles)
     % 获取popupmenu中的内容
    contents = cellstr(get(hObject, 'String'));
    selectedOption = get(handles.popupmenu1, 'Value');

    % 获取全局图像
    global img;

    % 确保图像已加载
    if isempty(img)
        msgbox('图像未加载，请加载图像后再试', '错误', 'error');
        return;
    end

    % 弹出输入框，让用户输入噪声强度
    prompt = {'请输入噪声强度（例如：0.02）:'};
    dlgtitle = '噪声强度设置';
    dims = [1 35];
    definput = {'0.02'};  % 默认值
    answer = inputdlg(prompt, dlgtitle, dims, definput);

    if isempty(answer)
        return; 
    end

    noiseStrength = str2double(answer{1});

    if isnan(noiseStrength) || noiseStrength < 0 || noiseStrength > 1
        msgbox('请输入一个有效的噪声强度值，范围为 0 到 1', '输入错误', 'error');
        return;
    end

   
    switch contents{selectedOption}
        case '高斯噪声'
            % 获取图像的大小
            [height, width, num_channels] = size(img);
            
            % 将图像转换为double类型，以便进行运算
            imgDouble = double(img);
            
            % 生成高斯噪声
            
            for c = 1:num_channels
                for i = 1:height
                    for j = 1:width
                        % 生成两个均匀分布的随机数
                        u1 = rand();
                        u2 = rand();
                        
                        % 盒子-穆勒变换，计算两个标准正态分布的随机数
                        z0 = sqrt(-2 * log(u1)) * cos(2 * pi * u2);
                        z1 = sqrt(-2 * log(u1)) * sin(2 * pi * u2);
                        
                        % 对噪声进行缩放，模拟指定的高斯噪声强度
                        noise = noiseStrength * z0;  % 只使用z0，z1可用于生成两个独立的噪声
                        
                        % 将噪声添加到图像
                        img(i, j, c) = imgDouble(i, j, c) + noise * 255;  % 乘以255使噪声适应图像的像素范围
                    end
                end
            end
            
            % 将噪声图像限制在0到255之间
            img = uint8(min(max(img, 0), 255));
            
            % 显示添加噪声后的图像
            axes(handles.axes1);  % 在指定的axes上显示图像
            imshow(img);
            title('高斯噪声');
    
         case '均匀噪声'        
            
            min_val = -noiseStrength * 255;  % 噪声的最小值
            max_val = noiseStrength * 255;   % 噪声的最大值
            
            % 分离彩色图像的RGB通道
            r = img(:,:,1);  % 红色通道
            g = img(:,:,2);  % 绿色通道
            b = img(:,:,3);  % 蓝色通道
            
            % 获取图像的大小
            [rows, cols] = size(r);
            
            % 手动生成均匀噪声并添加到每个通道
            noise_r = (max_val - min_val) * rand(rows, cols) + min_val;  % 生成红色通道的噪声
            r_noisy = double(r) + noise_r;  % 将噪声加到红色通道
            r_noisy = uint8(min(max(r_noisy, 0), 255));  % 限制像素值在 [0, 255] 范围内
            
            noise_g = (max_val - min_val) * rand(rows, cols) + min_val;  % 生成绿色通道的噪声
            g_noisy = double(g) + noise_g;  % 将噪声加到绿色通道
            g_noisy = uint8(min(max(g_noisy, 0), 255));  % 限制像素值在 [0, 255] 范围内
            
            noise_b = (max_val - min_val) * rand(rows, cols) + min_val;  % 生成蓝色通道的噪声
            b_noisy = double(b) + noise_b;  % 将噪声加到蓝色通道
            b_noisy = uint8(min(max(b_noisy, 0), 255));  % 限制像素值在 [0, 255] 范围内
            
            % 合并带噪声的RGB通道
            img = cat(3, r_noisy, g_noisy, b_noisy);            
            
            % 显示添加噪声后的图像
            axes(handles.axes1);  % 在指定的axes上显示图像
            imshow(img);
            title('均匀噪声');

        case '椒盐噪声'
            % 获取图像的大小
            [height, width, num_channels] = size(img);
            
            % 计算椒盐噪声的数量
            numSalt = round(noiseStrength * numel(img) / 2);  % 盐噪声数量
            numPepper = round(noiseStrength * numel(img) / 2);  % 椒噪声数量            
            
            % 随机生成盐噪声的位置
            saltIndices = rand(height, width, num_channels) < (numSalt / numel(img));
            % 随机生成椒噪声的位置
            pepperIndices = rand(height, width, num_channels) < (numPepper / numel(img));
            
            % 将盐噪声设置为白色（255）
            img(saltIndices) = 255;
            
            % 将椒噪声设置为黑色（0）
            img(pepperIndices) = 0;
            
            % 显示添加噪声后的图像
            axes(handles.axes1);  % 在指定的axes上显示图像
            imshow(img);
            title('椒盐噪声');          

    end

% --- Executes during object creation, after setting all properties.
function popupmenu1_CreateFcn(hObject, eventdata, handles)
    
    if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
        set(hObject,'BackgroundColor','white');
    end

    


% --- Executes on selection change in popupmenu2.
function popupmenu2_Callback(hObject, eventdata, handles)
     % 获取popupmenu中的内容
    contents = cellstr(get(hObject, 'String'));
    selectedOption = get(handles.popupmenu5, 'Value');
    
    % 获取全局图像
    global img;
    
    % 确保图像已加载
    if isempty(img)
        msgbox('图像未加载，请加载图像后再试', '错误', 'error');
        return;
    end
  
    switch contents{selectedOption}
        case '双边滤波'          
            img=im2double(img);           
            
            % 设置双边滤波参数
            d = 5;                % 邻域大小
            sigmaColor = 75;      % 色彩空间的标准差
            sigmaSpace = 75;      % 空间距离的标准差
            
            % 执行双边滤波
            imgFiltered= bilateralFilter(img, d, sigmaColor, sigmaSpace);
          
        case '均值滤波'
            imgFiltered = meanFilter(img, 3);  % 使用自定义的均值滤波函数
        case '平滑滤波(模糊技术)'
              
           imgFiltered = PM_Color(img, 5);
            
        otherwise
            return;
    end
    
    % 显示处理后的图像
    axes(handles.axes3);  % 在指定的axes上显示图像
    imshow(imgFiltered);
    title([contents{selectedOption}]);



% 自定义均值滤波函数
function output = meanFilter(inputImg, filterSize)
    [height, width, numChannels] = size(inputImg);
    padSize = floor(filterSize / 2);
    paddedImg = padarray(inputImg, [padSize, padSize], 'replicate');  % 填充边界
    output = zeros(height, width, numChannels, 'uint8');
    
    for c = 1:numChannels
        for i = 1:height
            for j = 1:width
                region = paddedImg(i:i+filterSize-1, j:j+filterSize-1, c);  % 获取邻域区域
                output(i, j, c) = uint8(mean(region(:)));  % 计算邻域区域的平均值
            end
        end
    end






function output = bilateralFilter(inputImage, d, sigmaColor, sigmaSpace)
    % inputImage: 输入图像
    % d: 邻域直径
    % sigmaColor: 色彩空间的标准差
    % sigmaSpace: 空间距离的标准差

    [rows, cols, channels] = size(inputImage);  % 获取图像的尺寸
    output = zeros(rows, cols, channels);       % 初始化输出图像

    % 计算空间高斯核
    [X, Y] = meshgrid(-d:d, -d:d);
    spatialGaussian = exp(-(X.^2 + Y.^2) / (2 * sigmaSpace^2));
    
    for c = 1:channels  % 遍历每个颜色通道
        currentChannel = inputImage(:, :, c);
        
        for i = 1:rows
            for j = 1:cols
                % 定义当前像素点周围的邻域区域                
                window = currentChannel(max(i-d, 1):min(i+d, rows), max(j-d, 1):min(j+d, cols));
                
                % 计算强度差异（对比度）的高斯权重
                intensityGaussian = exp(-((window - currentChannel(i,j)).^2) / (2 * sigmaColor^2));
                
                % 将空间高斯和强度高斯加权
                weight = spatialGaussian(1:size(window, 1), 1:size(window, 2)) .* intensityGaussian;
                
                % 归一化权重
                weight = weight / sum(weight(:));
                
                % 应用权重到邻域像素
                output(i, j, c) = sum(weight(:) .* window(:));
            end
        end
    end
    
    % 输出图像保持为 [0, 1] 范围
    output = max(0, min(1, output));



function imgFiltered = PM_Color(img, N)
    % 将彩色图像转换为 double 类型
    img = im2double(img); 
    
    % 分离 RGB 通道
    R = img(:,:,1);  % 红色通道
    G = img(:,:,2);  % 绿色通道
    B = img(:,:,3);  % 蓝色通道
    
    % 对每个通道应用平滑滤波
    R_filtered = PM(R, N);
    G_filtered = PM(G, N);
    B_filtered = PM(B, N);
    
    % 合并滤波后的通道
    imgFiltered = cat(3, R_filtered, G_filtered, B_filtered);


% PM 函数：基于模糊技术的平滑滤波
function newI = PM(I, N)
    [m, n] = size(I); % 获取图像的大小
    I = double(I);     % 转换为 double 类型
    newI = I;          % 初始化输出图像
    half_window = floor(N / 2); % 计算窗口的一半大小
    sNum = N^2 - 1;    % 窗口内像素的总数
    
    % 遍历每个像素
    for i = 1 + half_window : m - half_window
        for j = 1 + half_window : n - half_window
            newI(i, j) = result(I, i, j, half_window, sNum);
        end
    end
    
    % 转换回 uint8 类型
    newI = newI; % 无需缩放，直接返回 double 类型的值


% result 函数：计算给定像素的加权平均值
function pix = result(I, i, j, N, sNum)
    sum = 0;
    
    % 计算平方误差的和
    for m = i - N : i + N
        for n = j - N : j + N
            d = (I(i, j) - I(m, n))^2; % 像素值的平方差
            sum = sum + d; % 计算总误差
        end
    end
    
    % 计算 beta 值（用于高斯权重）
    beta = sum / sNum;
    sum2 = 0;
    sum3 = 0;
    
    % 计算加权平均值
    for m = i - N : i + N
        for n = j - N : j + N
            d = (I(i, j) - I(m, n))^2; % 计算像素间的差异
            mu = exp(-d / beta); % 计算高斯权重
            sum2 = sum2 + mu * I(m, n); % 权重加权后的像素值和
            sum3 = sum3 + mu; % 权重总和
        end
    end
    
    % 返回加权平均值
    pix = sum2 / sum3;



function popupmenu2_CreateFcn(hObject, eventdata, handles)

    if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
        set(hObject,'BackgroundColor','white');
    end


    



function popupmenu3_Callback(hObject, eventdata, handles)
    contents = cellstr(get(hObject, 'String'));
    selectedOption = get(handles.popupmenu3, 'Value');
    
    % 获取全局图像
    global img;
    global img2;
    
    % 确保图像已加载
    if isempty(img)
        msgbox('图像未加载，请加载图像后再试', '错误', 'error');
        return;
    end  
  % 将彩色图像转为灰度图像
    if size(img, 3) == 3
        img = rgb2gray(img);  % 将彩色图像转为灰度图
    end

    switch contents{selectedOption}
        
        case '灰度直方图'           
           
            % 调用函数计算手动计算的灰度直方图
            histogram = hist_img(img);
             % 在指定的轴上绘制灰度直方图
            axes(handles.axes3);  % 指定显示区域
            cla reset;  % 清空当前图形窗口
            bar(0:255, histogram, 'BarWidth', 0.5); 
            title('手动计算的灰度直方图');
            xlabel('灰度级');
            ylabel('像素数');
            xlim([0 255]);  
        case '直方图均衡化'          
            
            % 计算图像的直方图
            histgram = hist_img(img);
            
            [h, w] = size(img);  % 获取图像的高度和宽度
            new_img = zeros(h, w);  % 创建一个新图像
            
            % 初始化累积直方图
            s = zeros(256, 1);  
            s(1) = histgram(1);  
            for t = 2:256
                s(t) = s(t - 1) + histgram(t);
            end
            
            % 进行均衡化计算，并确保结果在 [0, 255] 范围内
            for x = 1:w
                for y = 1:h
                    new_img(y, x) = round(s(img(y, x) + 1) / (w * h) * 255);  % 映射到[0, 255]
                end
            end
            
            % 计算新的直方图
            newimg = hist_img(new_img);
            
            axes(handles.axes4);
            bar(0:255, newimg, 'BarWidth', 0.5);  
            title('直方图均衡化');
            xlabel('灰度级');
            ylabel('像素数');
            xlim([0 255]);
            
        case '直方图匹配'
        
            if size(img2, 3) == 3
                img2 = rgb2gray(img2);
            end
            
            % 计算源图像和参考图像的直方图
            hist_src = hist_img(img);
            hist_ref = hist_img(img2);
            
            % 计算源图像和参考图像的累积分布函数（CDF）
            cdf_src = cumsum(hist_src) / numel(img);  % 源图像的CDF
            cdf_ref = cumsum(hist_ref) / numel(img2);  % 参考图像的CDF
            
            % 构建映射表
            map = zeros(256, 1);
            for i = 1:256
                [~, idx] = min(abs(cdf_src(i) - cdf_ref));  % 找到最接近的值
                map(i) = idx - 1;
            end
            
            % 使用映射表生成新图像
            new_img = uint8(arrayfun(@(x) map(x + 1), img));  % 映射每个像素
            
            % 显示直方图规定化后的图像
            axes(handles.axes3);
            imshow(new_img);
            title('直方图规定化');
            
            % 计算并显示规定化后的直方图
            new_hist = hist_img(new_img);
            axes(handles.axes4);
            bar(0:255, new_hist, 'BarWidth', 0.5);
            title('直方图规定化后的直方图');
            xlabel('灰度级');
            ylabel('像素数');
            xlim([0 255]);
            
    end


% 定义函数：手动计算灰度直方图
function histogram = hist_img(img)
    % 初始化直方图数组，大小为256，表示灰度值从0到255
    histogram = zeros(1, 256);
    
    % 获取图像的大小
    [rows, cols] = size(img);
    
    % 遍历图像的每一个像素，并更新直方图
    for i = 1:rows
        for j = 1:cols
            gray_value = img(i, j);  % 获取当前像素的灰度值
            histogram(gray_value + 1) = histogram(gray_value + 1) + 1; 
        end
    end


function popupmenu3_CreateFcn(hObject, eventdata, handles)

    if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
        set(hObject,'BackgroundColor','white');
    end
% --- Executes on selection change in popupmenu4.
function popupmenu4_Callback(hObject, eventdata, handles)
    contents = cellstr(get(hObject, 'String'));
    selectedOption = get(handles.popupmenu4, 'Value');
    
    % 获取全局图像
    global img;
    
    % 确保图像已加载
    if isempty(img)
        msgbox('图像未加载，请加载图像后再试', '错误', 'error');
        return;
    end
   % 判断图像是否为彩色图像，若是则转为灰度图像
    if size(img, 3) == 3
         img_gray =  grayimg(img);  % 如果是彩色图像，先转换为灰度图像
    else
         img_gray = img;  % 如果已经是灰度图像，直接使用
    end

    switch contents{selectedOption}
        case '分段线性变换'
            img_double=im2double(img_gray);
            [h,w]=size(img_double);
            new_img=zeros(h,w);
            a=30/256;b=100/256;c=75/256;d=200/256;
            for x=1:w
                for y=1:h
                    if img_double(y,x)<a
                        new_img(y,x)=img_double(y,x)*c/a;
                    elseif img_double(y,x)<b
                        new_img(y,x)=(img_double(y,x)-a)*(d-c)/(b-a)+c;
                    else
                         new_img(y,x)=(img_double(y,x)-b)*(1-d)/(1-b)+d;
                    end
                   
                end
            end  
            axes(handles.axes3);
            cla reset;
            imshow(new_img);
            title('分段线性变换图像');
        case '窗切片处理'
            img_double=im2double(img_gray);
            [h,w]=size(img_double);
            new_img=zeros(h,w);
            a=170/256;b=200/256;c=90/256;d=250/256;
            for x=1:w
                for y=1:h
                    if img_double(y,x)<a
                        new_img(y,x)=c;
                    else
                        new_img(y,x)=d;
                    end
                end
            end
            axes(handles.axes3);
            cla reset;
            imshow(new_img);
            title('窗切片处理图像');
        case '对数变换'
            % 获取图像的大小
            [height, width] = size(img_gray);
            
            % 图像的最大像素值
            c = 255;
            
            % 创建一个空的图像数组用于存储对数变换后的结果
            img_log = zeros(height, width, 'uint8');
            
            % 对每个像素值进行对数变换
            for i = 1:height
                for j = 1:width
                    % 获取当前像素值
                    pixel_value = double(img_gray(i, j));
                    
                    % 应用对数变换公式                    
                    new_value = c * log(1 + pixel_value) / log(c + 1);
                    
                    % 确保值在 [0, 255] 范围内
                    img_log(i, j) = uint8(min(max(new_value, 0), 255));
                end
            end
            
           
            axes(handles.axes3);
            cla reset;
            imshow(img_log);
            title('对数变换后的图像');
        case '指数变换'
            % 获取图像的大小
            [height, width] = size(img_gray);
            
            % 常数 c 的设置，通常设置为 255
            c = 255;
            
            % 指数变换的参数 γ，设为大于 1 的值来增强图像的亮度
            gamma = 1.5;  % 你可以根据需要调整 γ 的值
            
            % 创建一个空的图像数组用于存储指数变换后的结果
            img_exp = zeros(height, width, 'uint8');
            
            % 对每个像素值进行指数变换
            for i = 1:height
                for j = 1:width
                    % 获取当前像素值
                    pixel_value = double(img_gray(i, j));
                    
                    % 应用指数变换公式
                    % 先归一化像素值到 [0, 1] 区间，然后进行指数变换，最后再映射回 [0, 255] 区间
                    new_value = c * (pixel_value / c) ^ gamma;
                    
                    % 确保值在 [0, 255] 范围内
                    img_exp(i, j) = uint8(min(max(new_value, 0), 255));
                end
            end
            
            % 在 axes2 上显示指数变换后的图像
            axes(handles.axes3);
            cla reset;
            imshow(img_exp);
            title('指数变换后的图像');
    end

function popupmenu4_CreateFcn(hObject, eventdata, handles)
    
    if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
        set(hObject,'BackgroundColor','white');
    end


    


% --- Executes on selection change in popupmenu5.
function popupmenu5_Callback(hObject, eventdata, handles)
    % 获取popupmenu中的内容
    contents = cellstr(get(hObject, 'String'));
    selectedOption = get(handles.popupmenu2, 'Value');
    
    % 获取全局图像
    global img;
    
    % 确保图像已加载
    if isempty(img)
        msgbox('图像未加载，请加载图像后再试', '错误', 'error');
        return;
    end
     if size(img, 3) == 3
         img_gray =  grayimg(img);  % 转换为灰度图像
     else
         img_gray = img;
     end
   
    switch contents{selectedOption}
        case 'robert'
            % Robert边缘检测           
            [height, width] = size(img_gray);
            Gx = [1 0; 0 -1];  % 水平梯度模板
            Gy = [0 1; -1 0];  % 垂直梯度模板
            edge_img = zeros(height, width, 'double');
            for i = 1:height-1
                for j = 1:width-1
                    region = double(img_gray(i:i+1, j:j+1));
                    grad_x = sum(sum(Gx .* region));
                    grad_y = sum(sum(Gy .* region));
                    magnitude = sqrt(grad_x^2 + grad_y^2);
                    edge_img(i, j) = magnitude;
                end
            end
            edge_img = uint8(edge_img);
            threshold = 100;
            edge_img(edge_img < threshold) = 0;
            edge_img(edge_img >= threshold) = 255;
            axes(handles.axes3);
            cla reset;
            imshow(edge_img);
            title([contents{selectedOption}]);
            
        case 'prewitt'
            % Prewitt边缘检测          
            [height, width] = size(img_gray);
            Gx = [1 0 -1; 1 0 -1; 1 0 -1];  % 水平梯度模板
            Gy = [1 1 1; 0 0 0; -1 -1 -1];  % 垂直梯度模板
            edge_img = zeros(height, width, 'double');
            for i = 2:height-1
                for j = 2:width-1
                    region = double(img_gray(i-1:i+1, j-1:j+1));
                    grad_x = sum(sum(Gx .* region));
                    grad_y = sum(sum(Gy .* region));
                    magnitude = sqrt(grad_x^2 + grad_y^2);
                    edge_img(i, j) = magnitude;
                end
            end
            edge_img = uint8(edge_img);
            threshold = 100;
            edge_img(edge_img < threshold) = 0;
            edge_img(edge_img >= threshold) = 255;
            axes(handles.axes3);
            cla reset;
            imshow(edge_img);
            title('Prewitt边缘检测');
            
        case 'sobel'
            % Sobel边缘检测
            [height, width] = size(img_gray);
            Gx = [-1 0 1; -2 0 2; -1 0 1];  % 水平梯度模板
            Gy = [-1 -2 -1; 0 0 0; 1 2 1];  % 垂直梯度模板
            edge_img = zeros(height, width, 'double');
            for i = 2:height-1
                for j = 2:width-1
                    region = double(img_gray(i-1:i+1, j-1:j+1));
                    grad_x = sum(sum(Gx .* region));
                    grad_y = sum(sum(Gy .* region));
                    magnitude = sqrt(grad_x^2 + grad_y^2);
                    edge_img(i, j) = magnitude;
                end
            end
            edge_img = uint8(edge_img);
            threshold = 100;
            edge_img(edge_img < threshold) = 0;
            edge_img(edge_img >= threshold) = 255;
            axes(handles.axes3);
            cla reset;
            imshow(edge_img);
            title('Sobel边缘检测');
            
        case '拉普拉斯'
            % 拉普拉斯边缘检测
            [height, width] = size(img_gray);
            L = [0 1 0; 1 -4 1; 0 1 0];
            edge_img = zeros(height, width, 'double');
            for i = 2:height-1
                for j = 2:width-1
                    region = double(img_gray(i-1:i+1, j-1:j+1));
                    laplacian_value = sum(sum(L .* region));
                    edge_img(i, j) = laplacian_value;
                end
            end
            edge_img = uint8(edge_img);
            threshold = 30;
            edge_img(edge_img < threshold) = 0;
            edge_img(edge_img >= threshold) = 255;
            axes(handles.axes3);
            cla reset;
            imshow(edge_img);
            title('拉普拉斯边缘检测');
    end
    

% --- Executes during object creation, after setting all properties.
function popupmenu5_CreateFcn(hObject, eventdata, handles)

if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end


% --- Executes on button press in pushbutton6.
function pushbutton6_Callback(hObject, eventdata, handles)
    global img;
    img =  grayimg(img);
    [N,M]=size(img);
    P=8;R=2;
    lbp=zeros(N,M);
    for j=2:N-1
        for i=2:M-1
            neighbor=[j-1 i-1;j-1 i;j-1 i+1;j i+1;j+1 i+1;j+1 i;j+1 i-1;j i-1];
            count=0;
            for k=1:8
                if img(neighbor(k,1),neighbor(k,2))>img(j,i)
                    count=count+2^(8-k);
                end 
            end 
            lbp(j,i)=count;
        end 
    end 
    lbp=uint8(lbp);   
    axes(handles.axes3);  % 在指定的axes上显示图像
    imshow(lbp);   
    title('lbp');


% --- Executes on button press in pushbutton7.
function pushbutton7_Callback(hObject, eventdata, handles)
    global img;
    img = double(rgb2gray(img)); % 读取并转换为灰度图像
    [N, M] = size(img);
    img = sqrt(img); % 进行开方操作

    % 定义 Sobel 算子进行边缘检测
    Hy = [-1 0 1]; Hx = Hy'; % Sobel 算子
    Gy = imfilter(img, Hy, 'replicate'); % 纵向梯度
    Gx = imfilter(img, Hx, 'replicate'); % 横向梯度
    Grad = sqrt(Gx.^2 + Gy.^2); % 梯度幅值

    % 计算相位角度
    Phase = zeros(N, M); Eps = 0.0001; % 初始化相位矩阵
    for i = 1:M
        for j = 1:N
            if abs(Gx(j, i)) < Eps && abs(Gy(j, i)) < Eps
                Phase(j, i) = 270; % 没有梯度的情况
            elseif abs(Gx(j, i)) < Eps && abs(Gy(j, i)) > Eps
                Phase(j, i) = 90; % 垂直边缘
            else
                Phase(j, i) = atan(Gy(j, i) / Gx(j, i)) * 180 / pi; % 计算梯度方向
                if Phase(j, i) < 0
                    Phase(j, i) = Phase(j, i) + 180; % 保证相位在[0, 180]区间内
                end
            end
        end
    end
    axes(handles.axes3);  % 在指定的axes上显示图像
    imshow(Phase);   
    title('hog');
    axes(handles.axes4);  % 在指定的axes上显示图像
    imshow(Grad);   
    title('hog2');
