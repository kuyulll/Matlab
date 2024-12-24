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


function GUI_OpeningFcn(hObject, eventdata, handles, varargin)
handles.output = hObject;

% Update handles structure
guidata(hObject, handles);

function varargout = GUI_OutputFcn(hObject, eventdata, handles) 
varargout{1} = handles.output;


function pushbutton1_Callback(hObject, eventdata, handles)
    global img;
    global path;
    [filename, pathname] = uigetfile('*.jpg;*.png;*.bmp', '选择图片');
    path = fullfile(pathname, filename);
    img = imread(path);
    axes(handles.axes1);
    cla reset;
    imshow(img);


function pushbutton2_Callback(hObject, eventdata, handles)
    global img2;
    [filename, pathname] = uigetfile('*.jpg;*.png;*.bmp', '选择图片');
    path = [pathname, filename];
    img2 = imread(path);
    axes(handles.axes2);
    cla reset;
    imshow(img2);

    

function pushbutton3_Callback(hObject, eventdata, handles)
    global img;
    gray_img=grayimg(img);
    axes(handles.axes3);
    cla reset;
    imshow(gray_img);
    title('灰度化图片');

function gray_img=grayimg(img)
     if size(img, 3) == 3
        R = double(img(:,:,1));
        G = double(img(:,:,2));
        B = double(img(:,:,3));    
        % 计算灰度图像
        gray_img = 0.299 * R + 0.587 * G + 0.114 * B;        
        gray_img = uint8(gray_img);
    end


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

    if isnan(scale_factor_height) || isnan(scale_factor_width)
        errordlg('请输入有效的数字作为缩放因子！', '错误'); % 输入无效则弹出错误对话框
        return;
    end

    [original_height, original_width, num_channels] = size(img);

    % 计算缩放后的尺寸
    new_height = round(original_height * scale_factor_height);
    new_width = round(original_width * scale_factor_width);
    img_resized = zeros(new_height, new_width, num_channels, 'uint8');

    % 最近邻插值：遍历每个新图像的像素，找到对应的原图像像素
    for i = 1:new_height
        for j = 1:new_width
            orig_i = round(i / scale_factor_height);
            orig_j = round(j / scale_factor_width);
            orig_i = min(max(orig_i, 1), original_height);
            orig_j = min(max(orig_j, 1), original_width);
            img_resized(i, j, :) = img(orig_i, orig_j, :);
        end
    end

    axes(handles.axes3);
    cla reset;
    imshow(img_resized);
    title(['缩放后的图片， 高度因子：', num2str(scale_factor_height), '，宽度因子：', num2str(scale_factor_width)]);



function pushbutton5_Callback(hObject, eventdata, handles)
     global img;
    
    % 输入的旋转角度
    prompt = '请输入旋转角度（度数）：'
    angle = str2double(inputdlg(prompt));     
    if isnan(angle)
        errordlg('请输入有效的数字角度！', '错误'); % 如果输入无效，显示错误信息
        return;
    end   

    theta = deg2rad(angle);
    [rows, cols, channels] = size(img);
    img_rotated = uint8(zeros(rows, cols, channels)); 
    
    % 计算图像中心
    center_x = floor(cols / 2);
    center_y = floor(rows / 2);
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
            if new_x > 0 && new_x <= cols && new_y > 0 && new_y <= rows
                img_rotated(y, x, :) = img(new_y, new_x, :);
            end
        end
    end    

    axes(handles.axes3);
    cla reset;  
    imshow(img_rotated);  
    title(['旋转角度：', num2str(angle), '°']); 



function popupmenu1_Callback(hObject, eventdata, handles)
    contents = cellstr(get(hObject, 'String'));
    selectedOption = get(handles.popupmenu1, 'Value');
    global img;
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
                        noise = noiseStrength * z0;                         
                        % 将噪声添加到图像
                        img(i, j, c) = imgDouble(i, j, c) + noise * 255;  % 乘以255使噪声适应图像的像素范围
                    end
                end
            end
            
            img = uint8(min(max(img, 0), 255));
            
            axes(handles.axes1); 
            cla reset;
            imshow(img);
            title('高斯噪声');
    
         case '均匀噪声'      
            min_val = -noiseStrength * 255;  % 噪声的最小值
            max_val = noiseStrength * 255;   % 噪声的最大值           
           
            r = img(:,:,1); 
            g = img(:,:,2); 
            b = img(:,:,3);           

            [rows, cols] = size(r);
            
            % 生成均匀噪声并添加到每个通道
            noise_r = (max_val - min_val) * rand(rows, cols) + min_val; 
            r_noisy = double(r) + noise_r;  
            r_noisy = uint8(min(max(r_noisy, 0), 255));
            
            noise_g = (max_val - min_val) * rand(rows, cols) + min_val; 
            g_noisy = double(g) + noise_g; 
            g_noisy = uint8(min(max(g_noisy, 0), 255));  
            
            noise_b = (max_val - min_val) * rand(rows, cols) + min_val;  
            b_noisy = double(b) + noise_b;  
            b_noisy = uint8(min(max(b_noisy, 0), 255));             
  
            img = cat(3, r_noisy, g_noisy, b_noisy);            
            
            axes(handles.axes1); 
            cla reset;
            imshow(img);
            title('均匀噪声');

        case '椒盐噪声'
            [height, width, num_channels] = size(img);
            
            % 计算椒盐噪声的数量
            numSalt = round(noiseStrength * numel(img) / 2); 
            numPepper = round(noiseStrength * numel(img) / 2);  
            
            % 随机生成盐噪声的位置
            saltIndices = rand(height, width, num_channels) < (numSalt / numel(img));
            % 随机生成椒噪声的位置
            pepperIndices = rand(height, width, num_channels) < (numPepper / numel(img));
            
            % 将盐噪声设置为白色（255）
            img(saltIndices) = 255;
            
            % 将椒噪声设置为黑色（0）
            img(pepperIndices) = 0;
            
            % 显示添加噪声后的图像
            axes(handles.axes1);  
            cla reset;
            imshow(img);
            title('椒盐噪声');          

    end


function popupmenu1_CreateFcn(hObject, eventdata, handles)    
    if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
        set(hObject,'BackgroundColor','white');
    end

    

function popupmenu2_Callback(hObject, eventdata, handles)
    contents = cellstr(get(hObject, 'String'));
    selectedOption = get(handles.popupmenu2, 'Value');
    
    % 获取全局图像
    global img;    
  
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
            imgFiltered = meanFilter(img, 3);  
        case '平滑滤波(模糊技术)'              
           imgFiltered = PM_Color(img, 5);
        case '梯度低通滤波'           
            if size(img, 3) == 3
                R = img(:,:,1);  
                G = img(:,:,2); 
                B = img(:,:,3);  
                channels = {R, G, B};  % 将RGB通道存储在一个cell数组中
            else               
                channels = {img, img, img};
            end           
            prompt = {'请输入最小截止频率 d0:', '请输入最大截止频率 d1:'};% 使用弹窗获取输入的截止频率 d0 和 d1
            dlgtitle = '输入截止频率';
            dims = [1 35]; 
            definput = {'5', '30'}; 
            answer = inputdlg(prompt, dlgtitle, dims, definput);            
            d0 = str2double(answer{1}); % 获取输入的d0和d1值并转化为数字
            d1 = str2double(answer{2});            
            % 计算频率坐标，避免每次循环都计算
            [N, M] = size(R);  
            [X, Y] = meshgrid(1:M, 1:N);
            X = X - floor(M / 2); % x轴偏移
            Y = Y - floor(N / 2); % y轴偏移            
            % 计算到频域中心的距离矩阵
            D = sqrt(X.^2 + Y.^2);            
            H = ones(N, M);              
            H(D > d1) = 0;  % 高于最大截止频率的部分为0
            H(D > d0 & D <= d1) = (D(D > d0 & D <= d1) - d1) / (d0 - d1); % 线性过渡区            
            filtered_channels = cell(1, 3);  % 存储滤波后的RGB通道            
            for j = 1:3               
                F = fftshift(fft2(double(channels{j})));  % 傅里叶变换
                F = F .* H;  % 应用滤波器
                g = ifftshift(F);  % 频谱移回原位置
                g = real(ifft2(g));  % 逆傅里叶变换到空域并取实部            
                % 存储滤波后的通道
                filtered_channels{j} = uint8(g);
            end            
            % 合并RGB通道
            imgFiltered = cat(3, filtered_channels{1}, filtered_channels{2}, filtered_channels{3});

        otherwise
            return;
    end   

    axes(handles.axes3);  
    cla reset;
    imshow(imgFiltered);
    title([contents{selectedOption}]);



% 均值滤波函数
function output = meanFilter(inputImg, filterSize)
    [height, width, numChannels] = size(inputImg);
    padSize = floor(filterSize / 2);
    paddedImg = padarray(inputImg, [padSize, padSize], 'replicate');  
    output = zeros(height, width, numChannels, 'uint8');
    
    for c = 1:numChannels
        for i = 1:height
            for j = 1:width
                region = paddedImg(i:i+filterSize-1, j:j+filterSize-1, c);  
                output(i, j, c) = uint8(mean(region(:)));  
            end
        end
    end

%双边滤波
function output = bilateralFilter(inputImage, d, sigmaColor, sigmaSpace)
    [rows, cols, channels] = size(inputImage); 
    output = zeros(rows, cols, channels);      

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

    output = max(0, min(1, output));



function imgFiltered = PM_Color(img, N)
    img = im2double(img);   

    R = img(:,:,1);  
    G = img(:,:,2);  
    B = img(:,:,3);  
    
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

    newI = newI; 


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
    global img;
    global img2;
    
    if isempty(img)
        msgbox('图像未加载，请加载图像后再试', '错误', 'error');
        return;
    end  
    if size(img, 3) == 3
        img = grayimg(img);  
    end

    switch contents{selectedOption}        
        case '灰度直方图'      
            histogram = hist_img(img);
            axes(handles.axes3);  
            cla reset;   
            bar(0:255, histogram, 'BarWidth', 0.5); 
            title('手动计算的灰度直方图');
            xlabel('灰度级');
            ylabel('像素数');
            xlim([0 255]);  
        case '直方图均衡化'        
            histgram = hist_img(img);            
            [h, w] = size(img);  
            new_img = zeros(h, w);         
            % 初始化累积直方图
            s = zeros(256, 1);  
            s(1) = histgram(1);  
            for t = 2:256
                s(t) = s(t - 1) + histgram(t);
            end            
            % 进行均衡化计算
            for x = 1:w
                for y = 1:h
                    new_img(y, x) = round(s(img(y, x) + 1) / (w * h) * 255);  % 映射到[0, 255]
                end
            end
            
            % 计算新的直方图
            newimg = hist_img(new_img);
            
            axes(handles.axes4);
            cla reset;
            bar(0:255, newimg, 'BarWidth', 0.5);  
            title('直方图均衡化');
            xlabel('灰度级');
            ylabel('像素数');
            xlim([0 255]);
            
        case '直方图匹配'        
            if size(img2, 3) == 3
                img2 =  grayimg(img2);
            end            
            % 计算源图像和参考图像的直方图
            hist_src = hist_img(img);
            hist_ref = hist_img(img2);            
            % 计算源图像和参考图像的累积分布函数（CDF）
            cdf_src = cumsum(hist_src) / numel(img); 
            cdf_ref = cumsum(hist_ref) / numel(img2);            
            % 构建映射表
            map = zeros(256, 1);
            for i = 1:256
                [~, idx] = min(abs(cdf_src(i) - cdf_ref)); 
                map(i) = idx - 1;
            end            
            % 使用映射表生成新图像
            new_img = uint8(arrayfun(@(x) map(x + 1), img));            
            % 显示直方图规定化后的图像
            axes(handles.axes3);
            cla reset;
            imshow(new_img);
            title('直方图规定化');            
           
            new_hist = hist_img(new_img);
            axes(handles.axes4);
            cla reset;
            bar(0:255, new_hist, 'BarWidth', 0.5);
            title('直方图规定化后的直方图');
            xlabel('灰度级');
            ylabel('像素数');
            xlim([0 255]);            
    end


% 计算灰度直方图
function histogram = hist_img(img)    
    histogram = zeros(1, 256);    
    [rows, cols] = size(img);  
    for i = 1:rows
        for j = 1:cols
            gray_value = img(i, j);  
            histogram(gray_value + 1) = histogram(gray_value + 1) + 1; 
        end
    end


function popupmenu3_CreateFcn(hObject, eventdata, handles)
    if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
        set(hObject,'BackgroundColor','white');
    end



function popupmenu4_Callback(hObject, eventdata, handles)
    contents = cellstr(get(hObject, 'String'));
    selectedOption = get(handles.popupmenu4, 'Value');   

    global img;
    
    if isempty(img)
        msgbox('图像未加载，请加载图像后再试', '错误', 'error');
        return;
    end

    if size(img, 3) == 3
         img_gray =  grayimg(img);  
    else
         img_gray = img;  
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
            [height, width] = size(img_gray);            
            % 图像的最大像素值
            c = 255;            
            % 创建一个空的图像数组用于存储对数变换后的结果
            img_log = zeros(height, width, 'uint8');            
            % 对每个像素值进行对数变换
            for i = 1:height
                for j = 1:width          
                    pixel_value = double(img_gray(i, j));                                     
                    new_value = c * log(1 + pixel_value) / log(c + 1);
                    img_log(i, j) = uint8(min(max(new_value, 0), 255));
                end
            end
            
            axes(handles.axes3);
            cla reset;
            imshow(img_log);
            title('对数变换后的图像');
        case '指数变换'
            [height, width] = size(img_gray);            
            % 常数 c 的设置，通常设置为 255
            c = 255;            
            % 指数变换的参数 γ，设为大于 1 的值来增强图像的亮度
            gamma = 1.5;              
            img_exp = zeros(height, width, 'uint8');            
            % 对每个像素值进行指数变换
            for i = 1:height
                for j = 1:width
                    pixel_value = double(img_gray(i, j));                    
                    % 应用指数变换公式
                    % 先归一化像素值到 [0, 1] 区间，然后进行指数变换，最后再映射回 [0, 255] 区间
                    new_value = c * (pixel_value / c) ^ gamma;                    
                    img_exp(i, j) = uint8(min(max(new_value, 0), 255));
                end
            end
            
            axes(handles.axes3);
            cla reset;
            imshow(img_exp);
            title('指数变换后的图像');
    end

function popupmenu4_CreateFcn(hObject, eventdata, handles)    
    if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
        set(hObject,'BackgroundColor','white');
    end


    
function popupmenu5_Callback(hObject, eventdata, handles)
    contents = cellstr(get(hObject, 'String'));
    selectedOption = get(handles.popupmenu5, 'Value');    

    global img;
    if isempty(img)
        msgbox('图像未加载，请加载图像后再试', '错误', 'error');
        return;
    end
     if size(img, 3) == 3
         img_gray =  grayimg(img);  
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
    


function popupmenu5_CreateFcn(hObject, eventdata, handles)

if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end


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
    axes(handles.axes3); 
    cla reset;
    imshow(lbp);   
    title('lbp');



function pushbutton7_Callback(hObject, eventdata, handles)
    global img;  
    grayImg = grayimg(img);     
    % 提取 HOG 特征
    hog_features = extractHOG(grayImg);    
    % 显示 HOG 特征的可视化
    axes(handles.axes3);
    cla reset;
    hold on;
    plot(hog_features);
    title('HOG Features Visualization');    

    [features, visualization] = extractHOGFeatures(grayImg);    
    axes(handles.axes4);
    cla reset;
    imshow(grayImg);
    hold on;
    plot(visualization);
    title('HOG Features Visualization');  
    

% 提取 HOG 特征的函数
function hog_features = extractHOG(grayImg)
    % 计算图像的梯度
    [Ix, Iy] = gradient(double(grayImg));
    magnitude = sqrt(Ix.^2 + Iy.^2);
    angle = atan2(Iy, Ix);
    
    % 设置 cell 大小
    cell_size = 8;
    [rows, cols] = size(grayImg);
    num_cells_x = floor(cols / cell_size);
    num_cells_y = floor(rows / cell_size);
    
    % 计算每个 cell 内的梯度方向直方图
    hog_features = zeros(num_cells_y, num_cells_x, 9);  
    bin_edges = linspace(-pi, pi, 10); 
    
    for i = 1:num_cells_y
        for j = 1:num_cells_x
            % 获取当前 cell 的区域
            x_start = (j-1) * cell_size + 1;
            x_end = j * cell_size;
            y_start = (i-1) * cell_size + 1;
            y_end = i * cell_size;            
            % 提取当前 cell 的梯度方向和幅值
            cell_magnitude = magnitude(y_start:y_end, x_start:x_end);
            cell_angle = angle(y_start:y_end, x_start:x_end);            
            % 计算梯度方向直方图
            for k = 1:numel(cell_magnitude)
                % 计算梯度方向所在的 bin
                bin_idx = find(cell_angle(k) >= bin_edges(1:end-1) & cell_angle(k) < bin_edges(2:end));
                if isempty(bin_idx)
                    bin_idx = 9;  % 处理 -pi 或 pi 边界的情况
                end
                hog_features(i, j, bin_idx) = hog_features(i, j, bin_idx) + cell_magnitude(k);
            end
        end
    end
    
    % 对 block 进行归一化
    block_size = 2;  % 2x2 cell
    num_blocks_x = num_cells_x - block_size + 1;
    num_blocks_y = num_cells_y - block_size + 1;
    normalized_hog = zeros(num_blocks_y, num_blocks_x, 36);  % 每个 block 由 4 个 cell 和 9 个 bins 组成
    
    for i = 1:num_blocks_y
        for j = 1:num_blocks_x
            % 提取 2x2 cells 的梯度直方图
            block_hist = hog_features(i:i+block_size-1, j:j+block_size-1, :);
            block_hist = block_hist(:);  % 展平为一个向量            
            % 对 block 进行 L2 归一化
            norm_factor = sqrt(sum(block_hist.^2) + 1e-6);  % 加上一个小常数避免除以零
            normalized_hog(i, j, :) = block_hist / norm_factor;
        end
    end
    hog_features = normalized_hog(:);





function pushbutton8_Callback(hObject, eventdata, handles)
    global path;   
    % 添加路径到 Python 环境
    py.sys.path().append('D:/matlab/MatlabGUI');   
    % 导入 Python 模块 'photo'
    py.importlib.import_module('photo');   
    model = 'D:/matlab/MatlabGUI/best.pt';   
    % 从 Python 获取预测结果
    try
        result = py.photo.img_pre(path, model);  % 获取预测结果        
        % 将 Python 字符串转换为 MATLAB 字符串
        resultStr = char(result);        
        % 如果结果为空或不可识别，显示 "无法识别的"
        if isempty(resultStr) || strcmp(resultStr, '')
            resultStr = '无法识别的';
        end
        
    catch
        % 如果发生错误，显示 "无法识别的"
        resultStr = '无法识别的';
    end  
    set(handles.text2, 'String', resultStr);



function pushbutton9_Callback(hObject, eventdata, handles)
    global img;
    grayImg = grayimg(img);
     histogram = hist_img(img);
     axes(handles.axes2); 
     cla reset; 
     bar(0:255, histogram, 'BarWidth', 0.5); 
     title('手动计算的灰度直方图');
     xlabel('灰度级');
     ylabel('像素数');
     xlim([0 255]);  
    
  
    % 弹窗输入阈值
    prompt = {'请输入阈值（0到255之间的值）:'};
    dlgtitle = '输入阈值';
    definput = {'0.3'};  % 默认值为 0.3
    answer = inputdlg(prompt, dlgtitle, [1 50], definput);
    
    % 获取输入的阈值并转换为数值
    threshold = str2double(answer{1}) * 255;    

    if isnan(threshold) || threshold < 0 || threshold > 255
        error('请输入一个有效的阈值（0到1之间）');
    end
    
    % 二值化函数
    binaryImg = customBinarize(grayImg, threshold);
    
    % 显示二值化图像   
    axes(handles.axes3); 
    cla reset;
    imshow(binaryImg);
    title('二值化图像');
    
    % 将前景部分置为黑色，背景保留
    foregroundBlack = img;
    
    % 将二值图像复制到3个通道
    binaryImg3D = customRepMat(binaryImg, 3);
    
    % 通过扩展后的binaryImg3D来处理前景与背景
    foregroundBlack(binaryImg3D == 1) = 0; % 前景置黑，保留背景  
    
    axes(handles.axes4); 
    cla reset;
    imshow(foregroundBlack);
    title('提取的目标图像');


    % 二值化函数
function binaryImg = customBinarize(grayImg, threshold) 
    [rows, cols] = size(grayImg);
    binaryImg = zeros(rows, cols);
    
    for i = 1:rows
        for j = 1:cols
            if grayImg(i, j) > threshold
                binaryImg(i, j) = 1;  % 白色
            else
                binaryImg(i, j) = 0;  % 黑色
            end
        end
    end

% customRepMat函数
function output = customRepMat(input, numTimes)    
    [rows, cols] = size(input);
    output = zeros(rows, cols, numTimes);    
    for i = 1:numTimes
        output(:,:,i) = input;  
    end
