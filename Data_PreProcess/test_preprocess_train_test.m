mm = matfile('./perspective/preprocessed2.mat','Writable',false);

m_train = matfile('./perspective/train.mat','Writable',true);
m_test = matfile('./perspective/test.mat','Writable',true);

nData = 207920;


m_train.RGB = uint8(zeros(227,227,3,3));
m_train.D = uint16(zeros(227,227,3));
m_train.Y = uint8(zeros(1));
m_test.RGB = uint8(zeros(227,227,3,3));
m_test.D = uint16(zeros(227,227,3));
m_test.Y = uint8(zeros(1));

idxTest = 1; idxTrain = 1;
for idx = 1:nData
    disp(idx);
    I_RGB = mm.RGB(:,:,:,idx);
    I_D = mm.D(:,:,idx);
    I_Y = mm.Y(idx,1);
    if mod(idx,5) == 0
        m_test.RGB(:,:,:,idxTest) = I_RGB;
        m_test.D(:,:,idxTest) = I_D;
        m_test.Y(idxTest,1) = I_Y;
        idxTest = idxTest + 1;
    else
        m_train.RGB(:,:,:,idxTrain) = I_RGB;
        m_train.D(:,:,idxTrain) = I_D;
        m_train.Y(idxTrain,1) = I_Y;
        idxTrain = idxTrain + 1;
    end
end