%                        Alireza Shafe'
% No matter how sophisticated a computer,"Coding" is the only way
% of communicating your thoughts with a computer.
%% PCA for both data sets A and B
% Now we don't consider labels ! 
% We just need to reduce the features so we can classify our samples
% much easier in the future with less features being considered.
clc
clear all
close all
Data_a = load('a.txt');
Data_b = load('b.txt');
Ma = mean(Data_a);
Mb = mean(Data_b);
[row,col] = size(Data_a);
for i = 1:row
 Data_a(i,:) = Data_a(i,:)-Ma;
end
[row,col] = size(Data_b);
for i = 1:row
 Data_b(i,:) = Data_b(i,:)-Mb;
end
Sa = cov(Data_a);
Sb = cov(Data_b);
[Ta,Ea] = eig(Sa);
[Tb,Eb] = eig(Sb);
[row,col] = size(Ea);
Ea = Ea*ones(row,1);
[row,col] = size(Eb);
Eb = Eb*ones(row,1);
Ta = double(Ta);
Tb = double(Tb); 
Ea = double(Ea);
Eb = double(Eb);
% ommiting unnecessary data 
FVa = [];
for i=1:length(Ea)
    index = find(max(Ea)==Ea);
    FVa = [FVa,Ta(:,index(1))];
    Ea(index) =[];
    Ta(:,index(1)) =[];
end
[row,col] = size(Data_a);
Train_Data_a = [];
for i = 1:row
 Train_Data_a(i,:) = Data_a(i,:)*FVa(:,1:2);
end
figure(1)
 plot(Train_Data_a(:,1),Train_Data_a(:,2),'.')
 title(['PCA on data A for 1st and 2nd component'],'Color','m')
%  repeating for B
 FVb = [];
for i=1:length(Eb)
    index = find(max(Eb)==Eb);
    FVb = [FVb,Tb(:,index(1))];
    Eb(index) =[];
    Tb(:,index(1)) =[];
end
[row,col] = size(Data_b);
Train_Data_b = [];
for i = 1:row
 Train_Data_b(i,:) = Data_b(i,:)*FVb(:,1:2);
end
figure(2) 
plot(Train_Data_b(:,1),Train_Data_b(:,2),'.')
 title(['PCA on data B for 1st and 2nd component'],'Color','m')
% *As a result, classification will be done much easier for this Data*
%% 3 
%  for 3rd and 4th component
[row,col] = size(Data_a);
Train_Data_a = [];
for i = 1:row,
 Train_Data_a(i,:) = Data_a(i,:)*FVa(:,4:5);
end
figure(3)
plot(Train_Data_a(:,1),Train_Data_a(:,2),'.')
 title(['PCA on data A for 3rd and 4th component'],'Color','m')
[row,col] = size(Data_b);
Train_Data_b = [];
for i = 1:row
 Train_Data_b(i,:) = Data_b(i,:)*FVb(:,4:5);
end
figure(4)
plot(Train_Data_b(:,1),Train_Data_b(:,2),'.')
 title(['PCA on data B for 3rd and 4th component'],'Color','m')
%  for 9th and 10th component
[row,col] = size(Data_a);
Train_Data_a = [];
for i = 1:row
 Train_Data_a(i,:) = Data_a(i,:)*FVa(:,9:10);
end
figure(5)
plot(Train_Data_a(:,1),Train_Data_a(:,2),'.')
 title(['PCA on data A for 9th and 10th component'],'Color','m')
[row,col] = size(Data_b);
Train_Data_b = [];
for i = 1:row
 Train_Data_b(i,:) = Data_b(i,:)*FVb(:,9:10);
end
figure(6)
plot(Train_Data_b(:,1),Train_Data_b(:,2),'.')
 title(['PCA on data B for 9th and 10th component'],'Color','m')
 %% 4
%  [row,col] = size(Train _Data_b); 
%  rawData_b = []; 
%  for i = 1:row
%      rawData_b(i,:) = Train_Data_b(i,:)*FVb(:,1:2)';
%  end
[row,col] = size(Train_Data_a);
rawData_a = [];
for i = 1:row
rawData_a(i,:) =  Train_Data_a(i,:)*FVa(:,1:2)';
end
Variance_of_Data=var(rawData_a);
Variance_of_first_2_principals_Data=var(Train_Data_a);
Variance_of_Data=var(Variance_of_Data)
Variance_of_first_2_principals_Data=var(Variance_of_first_2_principals_Data)
% We can see how much of the variance have been retained.
% There is a reduction in Variance of Data.
%% 5
%  Here we consider classes
Data = xlsread('a1.xlt');
[row,col] = size(Data);
C1 =[];
C2 = [];
C3 =[];
% again we need to seperate our classes..
for i =1:row
    if Data(i,1)==1
        C1 = [C1;Data(i,:)];
    elseif Data(i,1)==2
        C2 = [C2;Data(i,:)];
    else
        C3 = [C3;Data(i,:)];
    end
end

M1 = mean(C1(:,2:end));
M2 = mean(C2(:,2:end));
M3 = mean(C3(:,2:end));
M = mean(Data(:,2:end));

S1 = cov(C1(:,2:end));
S2 = cov(C2(:,2:end));
S3 = cov(C3(:,2:end));

Sw = S1+S2+S3;
Sb = (M1-M)'*(M1-M)+(M2-M)'*(M2-M)+(M3-M)'*(M3-M);

W = Sw*Sb';
Sb = W;
% T=> eigenvector , E => eigenvalue
[Tb,Eb] = eig(Sb);
[row,col] = size(Eb);
Eb = Eb*ones(row,1);
Tb = double(Tb);
Eb = double(Eb);
FVb = [];
for i=1:length(Eb)
    index = find(max(Eb)==Eb);
    FVb = [FVb,Tb(:,index(1))];
    Eb(index) =[];
    Tb(:,index(1)) =[];
end
[row,col] = size(C1);
Train_Data_b = [];
for i = 1:row
 Train_Data_b(i,:) = C1(i,2:end)*FVb(:,1:2);
end
figure(7)
plot(Train_Data_b(:,1),Train_Data_b(:,2),'.')
 title(['Dimension Reduction using Fisher discriminant Analysis for Class 1'],'Color','m')

[row,col] = size(C2);
Train_Data_b = [];
for i = 1:row
 Train_Data_b(i,:) = C2(i,2:end)*FVb(:,1:2);
end
figure(8)
plot(Train_Data_b(:,1),Train_Data_b(:,2),'.')
 title(['Dimension Reduction using Fisher discriminant Analysis for Class 2'],'Color','m')

[row,col] = size(C3);
Train_Data_b = [];
for i = 1:row
 Train_Data_b(i,:) = C3(i,2:end)*FVb(:,1:2);
end
figure(9)
plot(Train_Data_b(:,1),Train_Data_b(:,2),'.')
 title(['Dimension Reduction using Fisher discriminant Analysis for Class 3'],'Color','m')