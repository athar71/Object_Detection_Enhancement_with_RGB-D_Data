function [IMAGE] = ImageResizeDepth(image,final_size,pad_value)
f_d = 592.5;

[h,w,c] = size(image);

if c==3
    dummy = uint8(ones(final_size,final_size,c)*pad_value);
elseif c==1
    dummy = uint16(ones(final_size,final_size,c)*pad_value);
end

if h>w
    scale = final_size/h;
else
    scale = final_size/w;
end

VecDist = zeros(w*h-1,1);
Counter = 1;

hC = floor(h/2);
wC = floor(w/2);
% dMid = image(hC,wC);
for idxH = 1:h
    for idxW = 1:w
        d = image(idxH,idxW);
        if d==0 || (idxH==hC && idxW==wC)
            continue
        else
            ratio = 1/f_d * sqrt( (idxW-wC)^2 + (idxH-hC)^2 );
            theta = asin(ratio);
            d_new = d*ratio/(sin(scale*theta));
            image(idxH,idxW) = d_new;
            VecDist(Counter) = d_new*cos(scale*theta) - d*cos(theta);
            Counter = Counter + 1;
        end
    end
end
image(hC,wC) = image(hC,wC) - mean(VecDist(VecDist~=0));

if h>w
    newImage = imresize(image,[final_size, floor(w*final_size/h)]);
    [~,w_new,~] = size(newImage);
    diff = floor((final_size-w_new)/2);
    dummy(:,diff+1:diff+w_new,:) = newImage;
    IMAGE = dummy;
elseif w>h
    newImage = imresize(image,[floor(h*final_size/w), final_size]);
    [h_new,~,~] = size(newImage);
    diff = floor((final_size-h_new)/2);
    dummy(diff+1:diff+h_new,:,:) = newImage;
    IMAGE = dummy;
else
    IMAGE = imresize(image,[final_size, final_size]);
end

end