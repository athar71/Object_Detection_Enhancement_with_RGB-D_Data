mm = matfile('./perspective/test.mat','Writable',false);

m_s = matfile('./perspective/test_s.mat','Writable',true);

nData = 41584;
% nData = 166336;


m_s.RGB = uint8(zeros(227,227,3,3));
m_s.D = uint16(zeros(227,227,3));
m_s.Y = uint8(zeros(1));

Vec = randperm(nData);

for idx = 1:nData
    disp(idx);
    IDX = Vec(idx);
    m_s.RGB(:,:,:,idx) = mm.RGB(:,:,:,IDX);
    m_s.D(:,:,idx) = mm.D(:,:,IDX);
    m_s.Y(idx,1) = mm.Y(IDX,1);
    
end