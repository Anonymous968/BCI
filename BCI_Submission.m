clc;
clear all;
close all;

signal=load('C:\Chanakya\Textbooks\Semester-III\Project Seminar\Subject1_2D(2).mat');
Fs=1000;

% Getting all the left hand data
left1(:,:,1)=signal.LeftBackward1.';
left1(:,:,2)=signal.LeftBackward2.';
left1(:,:,3)=signal.LeftBackward3.';
left1(:,:,4)=signal.LeftForward1.';
left1(:,:,5)=signal.LeftForward2.';
left1(:,:,6)=signal.LeftForward3.';

% Getting all the right hand data
right1(:,:,1)=signal.RightBackward1.';
right1(:,:,2)=signal.RightBackward2.';
right1(:,:,3)=signal.RightBackward3.';
right1(:,:,4)=signal.RightForward1.';
right1(:,:,5)=signal.RightForward2.';
right1(:,:,6)=signal.RightForward3.';

Samples=1002;

% Processing it to get 18 samples
for i=1:6
    for j=1:3
        left(:,:,i,j)=left1(:,(j-1)*Samples+1:j*Samples,i);
        right(:,:,i,j)=right1(:,(j-1)*Samples+1:j*Samples,i);
    end
end

% Performing the fft of each of the 18 samples
for i=1:6
    for j=1:3
        for k=1:19
            left_fft(k,:,i,j)=fft(left(k,:,i,j));
            right_fft(k,:,i,j)=fft(right(k,:,i,j));
        end
    end
end

left_filtered=zeros(19,4,6,3,4);
right_filtered=zeros(19,4,6,3,4);

M=zeros(4,2);

% Feature Extraction
for i=1:6
    for j=1:3
        for k=1:19
            [left_filtered(k,1,i,j,1),left_filtered(k,1,i,j,2)]=max(abs(left_fft(k,1:50,i,j)));
            [left_filtered(k,2,i,j,1),left_filtered(k,2,i,j,2)]=max(abs(left_fft(k,51:200,i,j)));
            [left_filtered(k,3,i,j,1),left_filtered(k,3,i,j,2)]=max(abs(left_fft(k,201:400,i,j)));
            [left_filtered(k,4,i,j,1),left_filtered(k,4,i,j,2)]=max(abs(left_fft(k,401:502,i,j)));
            left_filtered(k,2,i,j,2)=50+left_filtered(k,2,i,j,2);
            left_filtered(k,3,i,j,2)=200+left_filtered(k,3,i,j,2);
            left_filtered(k,4,i,j,2)=400+left_filtered(k,4,i,j,2);
            left_filtered(k,1,i,j,3)=angle(left_fft(k,left_filtered(k,1,i,j,2),i,j));
            left_filtered(k,2,i,j,3)=angle(left_fft(k,left_filtered(k,2,i,j,2),i,j));
            left_filtered(k,3,i,j,3)=angle(left_fft(k,left_filtered(k,3,i,j,2),i,j));
            left_filtered(k,4,i,j,3)=angle(left_fft(k,left_filtered(k,4,i,j,2),i,j));
            
            [right_filtered(k,1,i,j,1),right_filtered(k,1,i,j,2)]=max(abs(right_fft(k,1:50,i,j)));
            [right_filtered(k,2,i,j,1),right_filtered(k,2,i,j,2)]=max(abs(right_fft(k,51:200,i,j)));
            [right_filtered(k,3,i,j,1),right_filtered(k,3,i,j,2)]=max(abs(right_fft(k,201:400,i,j)));
            [right_filtered(k,4,i,j,1),right_filtered(k,4,i,j,2)]=max(abs(right_fft(k,401:502,i,j)));
            right_filtered(k,2,i,j,2)=50+right_filtered(k,2,i,j,2);
            right_filtered(k,3,i,j,2)=200+right_filtered(k,3,i,j,2);
            right_filtered(k,4,i,j,2)=400+right_filtered(k,4,i,j,2);
            right_filtered(k,1,i,j,3)=angle(right_fft(k,right_filtered(k,1,i,j,2),i,j));
            right_filtered(k,2,i,j,3)=angle(right_fft(k,right_filtered(k,2,i,j,2),i,j));
            right_filtered(k,3,i,j,3)=angle(right_fft(k,right_filtered(k,3,i,j,2),i,j));
            right_filtered(k,4,i,j,3)=angle(right_fft(k,right_filtered(k,4,i,j,2),i,j));
        end
    end
end

left_filter=reshape(left_filtered,19,4,18,4);
right_filter=reshape(right_filtered,19,4,18,4);

% Normalization
for i=1:19
    for j=1:3
        left_filter(i,:,:,j)=left_filter(i,:,:,j)./max(max(left_filter(i,:,:,j)));
        right_filter(i,:,:,j)=right_filter(i,:,:,j)./max(max(left_filter(i,:,:,j)));
    end
end

for i=1:19
    for j=1:18
        q=1;
        for l=1:4
            for m=1:3
                left_filter_comb(i,q,j)=left_filter(i,l,j,m);
                right_filter_comb(i,q,j)=right_filter(i,l,j,m);
                q=q+1;
            end
        end
    end
end

% Saving the files to a particular location so that the Keras Model can be
% trained
save('BCI_left_filter.m','left_filter_comb','-v7');
save('BCI_right_filter.m','right_filter_comb','-v7');
