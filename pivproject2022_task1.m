%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%          Instituto Superior Técnico       %
%-------------------------------------------%
%       Mestrado em Controlo, Robótica e    %
%           Inteligência Artificial         %
%-------------------------------------------%
% - Processamento de Imagem e Visão (PIV)   %
% - Projeto, Part I:                        %
% - Realizado por:                          %
%       - André Pires                       %
%       - Guilherme Fernandes               %
%       - José Mateus                       %
%       - Afonso Figueiredo                 %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%
% octave pivproject2022_task1.m \tmp \rgbcamera \H_transform

function pivproject2022_task1()

close all;
clear;

args = argv();
% args{1} =  "path_to_template"
template_dir = args{1};
% args{2} =  "path_to_input_folder"
input_dir = args{2};
% args{3} = "path_to_output_folder"
output_dir = args{3};

% Load template values

temp_path = strcat('**', template_dir, '/*.jpg');
temp_features_path = strcat('**', template_dir, '/*.mat');

disp(temp_path);
disp(temp_features_path);

template_pixels = load(dir(temp_features_path).name).p.';
template_descriptors = load(dir(temp_features_path).name).d.';
full_template = imread(dir(temp_path).name);

% temp_path = strcat('.', template_dir, '/templateSNS.jpg');
% temp_features_path = strcat('.', template_dir, '/features.mat');
% 
% template_pixels = load(temp_features_path).p.';
% template_descriptors = load(temp_features_path).d.';
% full_template= imread(temp_path);

% Load input values

img_path = strcat('**', input_dir, '/*.jpg');
img_features_path = strcat('**', input_dir, '/*.mat');

image_pixels = load(dir(img_features_path).name).p.';
image_descriptors = load(dir(img_features_path).name).d.';
full_image = imread(dir(img_path).name);

% img_path = strcat('.', input_dir, '/rgb_0001.jpg');
% img_features_path = strcat('.', input_dir, '/rgbsift_0001.mat');
% 
% image_pixels = load(img_features_path).p.';
% image_descriptors = load(img_features_path).d.';
% full_image = imread(img_path);

N_iterations=72;
inlier_error = 10;

good_matches = nearest_neighbours(image_descriptors, template_descriptors);

H_final = Ransac(image_pixels(good_matches(:,1),:), template_pixels(good_matches(:,2),:), N_iterations,inlier_error);

% Homography path folder
H_path = strcat('.', output_dir, '/H_0001.mat');

save(H_path, 'H_final');

end

function good_matches = nearest_neighbours(image_descriptors, template_descriptors)  

    correspondences = zeros(size(image_descriptors, 1), 2); 
    distances = zeros(size(image_descriptors, 1), 2); 
    for i = 1:size(image_descriptors, 1)  

         current_distances = sqrt(sum((template_descriptors - image_descriptors(i, :)) .^ 2, 2));           
         [sorted_distances, indices] = sort(current_distances, 1);
         correspondences(i, :) = indices(1:2).'; 
         distances(i, :) = sorted_distances(1:2).'; 
    
    end 
    n_good_matches=0;

    good_matches=[];
    for i = 1:length(image_descriptors)
        if distances(i,1) < 0.7*distances(i,2)
            n_good_matches = n_good_matches+1;
            good_matches(end+1,:) = [i,correspondences(i,1)];        
        end
    end
end

function H_final=Ransac(image_pixels,template_pixels,N_iterations,inlier_error)
    image_pixels_homogeneous=[image_pixels,ones(size(image_pixels,1),1)];  
    Maximum_inliers= [];
    
    for Iteration = 1:N_iterations
        current_inliers=[];
        
        %Chooses points used to build model 
        model_points = randperm(length(image_pixels),4);
        %builds model (Projection matrix) 
        
        if size(unique(template_pixels(model_points,:),'rows'),1)<4
            N_iterations=N_iterations+1;
            continue
        end
        
        Hmatrix=find_Hmatrix(image_pixels_homogeneous,template_pixels,model_points);
        %Hmatrix=homography_solve(image_pixels(model_points,:).',template_pixels(model_points,:).');
        %applies projection to all featu
        % res in image
        image_pixels_in_template_homogeneous= Hmatrix * image_pixels_homogeneous.';
        image_pixels_in_template = (image_pixels_in_template_homogeneous(1:2,:)./ image_pixels_in_template_homogeneous(3,:)).';       
        %Counts number of inliers with these projection
        
        %tform=fitgeotrans(image_pixels(model_points,:),template_pixels(model_points,:),"projective");
        %image_pixels_in_template = transformPointsForward(tform,image_pixels);
        for i = 1:length(image_pixels)
            if norm(template_pixels(i,:)-image_pixels_in_template(i,:))<inlier_error
                current_inliers(end+1) = i;        
            end
        end
        %If it has more inliers becomes best model so far
        if length(current_inliers)>length(Maximum_inliers)
            Maximum_inliers=current_inliers;
        end
    end
    %Uses biggest ammount of inliers found to have a better estimate of
    %the projection matrix
    if length(Maximum_inliers)<6
        msg = 'Was only able to find less than 6 inliers.';
        error(msg)
    end
    %tform=fitgeotrans(image_pixels(Maximum_inliers,:),template_pixels(Maximum_inliers,:),"projective");
    H_final=find_Hmatrix(image_pixels_homogeneous,template_pixels,Maximum_inliers);
    %H_final=homography_solve(image_pixels(Maximum_inliers,:).',template_pixels(Maximum_inliers,:).');

end

function Hmatrix=find_Hmatrix(image_pixels_homogeneous,template_pixels,model_points)
    A=zeros(length(model_points),9);
    i=1;
    for model_point=model_points
         A(i,:)=[-image_pixels_homogeneous(model_point,:),zeros(1,3),image_pixels_homogeneous(model_point,:).*template_pixels(model_point,1)];
        A(i+1,:)=[zeros(1,3),-image_pixels_homogeneous(model_point,:),image_pixels_homogeneous(model_point,:).*template_pixels(model_point,2)];
        i = i + 2;
    end

    [~,~,V] = svd(A,"econ") ;
    %Makes sure points are not linearly dependent
    H = V(:,end);
    Hmatrix = reshape(H,[3,3]).';
    Hmatrix=Hmatrix./Hmatrix(3,3);   
    
end
