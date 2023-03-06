function pivproject2022_task2()

    args = argv();

    % args{1} =  "ref_frame"
    ref_frame = "3";
    % args{2} =  "path_to_input_folder"
    input_dir = "\Quarto";
    % args{3} = "path_to_output_folder"
    output_dir = "\Quarto_out";

    % your code
    H = second_part(ref_frame, input_dir);

    for i = 1:length(H)
        Hi = H(i);
        Hi_name = strcat(output_dir,'H_', num2str(i,'%04.f'), '.mat');
        save(Hi_name, 'Hi')
    end
    display(H, input_dir)
end

function [h_i1] = second_part(ref_frame,input_dir)

    N_iterations = 72;
    inlier_error = 5;

    min_matches = 10;
    
    input = dir(input_dir);
    
    imgs = {};
    keypoints = {};
    
    for i = 1: length(input)
       file_name = input(i).name;
       if length(file_name) > 4
           file =  input(i).name(end-3:end);
           if strcmp(file ,'.jpg') || strcmp(file ,'.png') 
                imgs{end+1} = imread(strcat(input(i).folder,'/',file_name));
           elseif strcmp(file ,'.mat') 
                keypoints{end+1} = load(strcat(input(i).folder,'/',file_name));
           end
       end
    end

    nImages = length(input);
    
    % cell array of projection matrices 
    h_i1 = cell(1,nImages);
    
    % projection matrix for reference image is identity
    h_i1(ref_frame) = {eye(3)};
    
    % vector of images indexs to add to panorama images
    index_images = 1:nImages;
    index_images(ref_frame) = [];
    
    % matching failures
    match_fail = [];
    
    % vector of images indexs that have a projection to the panorama
    panorama_images = ref_frame;

    disp(panorama_images) %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
    index = 1;
    while true
        new_image = index_images(index);
        h_i1(new_image) = {zeros(3)};
        
        disp(panorama_images) %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

        for already_added_image = panorama_images
            
            disp(already_added_image) %%%%%%%%%%%%%%%%%%%%%%%%%%%%
            disp(panorama_images)
            
            % initicialize image to match panorama image     
            image_descriptors = keypoints{1,new_image}.d.';    
            template_descriptors = keypoints{1,already_added_image}.d.';
            
            % obtain good matches for this match
            good_matches = nearest_neighbours(image_descriptors, template_descriptors);

            if length(good_matches) > min_matches 

                image_pixels = load(keypoints.Files{new_image}).p.';
                template_pixels=load(keypoints.Files{already_added_image}).p.';
        
                % run RANSAC to get homography to the panorama image it matches
                h_ij = Ransac(image_pixels(good_matches(:,1),:), template_pixels(good_matches(:,2),:), N_iterations, inlier_error);
                if h_ij == zeros(3)
                    continue;
                end
                
                panorama_images{end+1} = new_image;
                index_images(index) = [];
                
                % obtain full homography in relation to reference image
                h_i1(new_image) = {h_ij * cell2mat(h_i1(already_added_image))};
             
                new_found = true;
                break;
            end
            
            if new_found
                index = 1;
            else
                index = index + 1;
            end
            
        end            
    end
end

function H_final=Ransac(image_pixels,template_pixels,N_iterations,inlier_error)
    image_pixels_homogeneous=[image_pixels,ones(size(image_pixels,1),1)];  
    Maximum_inliers= [];
    
    for Iteration = 1:N_iterations
        current_inliers=[];
        
        % Chooses points used to build model 
        model_points = randperm(length(image_pixels),4);

        % builds model (Projection matrix) 
        if size(unique(template_pixels(model_points,:),'rows'),1)<4
            N_iterations=N_iterations+1;
            continue
        end
        Hmatrix=find_Hmatrix(image_pixels_homogeneous,template_pixels,model_points);

        % applies projection to all features in image
        image_pixels_in_template_homogeneous= Hmatrix * image_pixels_homogeneous.';
        image_pixels_in_template = (image_pixels_in_template_homogeneous(1:2,:)./ image_pixels_in_template_homogeneous(3,:)).'; 

        % Counts number of inliers with these projection
        for i = 1:length(image_pixels)
            if norm(template_pixels(i,:)-image_pixels_in_template(i,:))<inlier_error
                current_inliers(end+1) = i;        
            end
        end

        % If it has more inliers becomes best model so far
        if length(current_inliers)>length(Maximum_inliers)
            Maximum_inliers=current_inliers;
        end
    end

    % Uses biggest ammount of inliers found to have
    % a better estimate of the projection matrix
    if length(Maximum_inliers)<6
        msg = 'Was only able to find less than 6 inliers.';
        error(msg)
    end

    H_final=find_Hmatrix(image_pixels_homogeneous,template_pixels,Maximum_inliers);

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

