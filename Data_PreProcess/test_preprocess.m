clc
clear


m = matfile('./perspective/preprocessed2.mat','Writable',true);

m.RGB = uint8(zeros(227,227,3,3));
m.D = uint16(zeros(227,227,3));
m.Y = uint8(zeros(1));
IdxImage = 1;

datasetDir = '../dataset/rgbd-dataset';
List = dir(datasetDir);
List(1:2) = []; % Remove . and ..

for idxDir = 1:length(List) % All the objects in the dataset
    label = idxDir;
    %     disp(label);
    subFolder = strcat(datasetDir, '/', List(idxDir).name);
    ListObjFolders = dir(subFolder);
    ListObjFolders(1:2) = []; % Remove . and ..
    for idxSubdir = 1:length(ListObjFolders) % All the folders of a single obj
        disp(num2str(label) + "_" + num2str(idxSubdir));
        subsubFolder = strcat(subFolder,'/',ListObjFolders(idxSubdir).name);
        ListObj = dir(strcat(subsubFolder,'/*depthcrop.png'));
        
        nObjectsInFolder = length(ListObj);
        for idx = 1:nObjectsInFolder
            strD = ListObj(idx).name;
            STRsplit = strsplit(strD,'_');
            STRsplit{end} = 'crop.png';
            strRGB = strjoin(STRsplit,'_');
            disp(strRGB);
            
            
            if ~isfile(strcat(subsubFolder,'/',strRGB))
                disp("Nonexisting RGB image");
                continue
            end
            ImageRGB = imread( strcat(subsubFolder,'/',strRGB) );
            ImageD = imread( strcat(subsubFolder,'/',strD) );
            
            IRGB = ImageResize(ImageRGB,227,255);
            ID = ImageResizeDepth(ImageD,227,0);
            
            m.RGB(:,:,:,IdxImage) = IRGB;
            m.D(:,:,IdxImage) = ID;
            m.Y(IdxImage,1) = uint8(label);
            IdxImage = IdxImage + 1;
        end
        
    end
    
end
