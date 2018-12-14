function [IMAGE] = ImageResize(image,final_size,pad_value)

[h,w,c] = size(image);

if c==3
    dummy = uint8(ones(final_size,final_size,c)*pad_value);
elseif c==1
    dummy = uint16(ones(final_size,final_size,c)*pad_value);
end

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