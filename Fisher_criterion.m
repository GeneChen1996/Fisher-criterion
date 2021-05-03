%%
clc;       % �M��command window
clear      % �M��workspace
close all  % �����Ҧ�figure

%% Ū��.txt���
dataSet = load('iris.txt');
rawData = dataSet(:,1:4);    % ��l��ơA75����� x 4�ӯS�x
label   = dataSet(:,5);      % 75����Ʃҹ���������
kk = 3; %knn k=3

trnset =[rawData(1:25,:);...
          rawData(51:75,:);...
          rawData(101:125,:)];
      
tstset =[rawData(26:50,:);...
          rawData(76:100,:);...
          rawData(126:150,:)];
      
%% Sw and Sb

c = [trnset(1:25,:),trnset(26:50,:),trnset(51:75,:)];


Sw =0;
Prior = 25/75;
for c =1:3
    if c ==1 
        Xij = trnset(1:25,:);       
    end  
    
    if c == 2
        Xij = trnset(26:50,:);
    end    
    
    if c == 3
        Xij = trnset(51:75,:);              
    end
    
    classmean = mean(Xij);
    count = (((Xij-classmean)'*(Xij-classmean))/25)*Prior;
    Sw = Sw+count;
end        


Sb = 0;

for c =1:3
    if c ==1 
        Xij = trnset(1:25,:);    
    end  
    
    if c == 2
        Xij = trnset(26:50,:);
    end    
    
    if c == 3
        Xij = trnset(51:75,:);              
    end
    meanj = mean(Xij);
    allclassesmean = mean(trnset);
    count2 = Prior*((meanj-allclassesmean)'*(meanj-allclassesmean));
    Sb = Sb+count2;
end

%% Fisher's score
Fi = Sb/Sw;
Fi = [Fi(1,1),Fi(2,2),Fi(3,3),Fi(4,4);];
[Firank,index] = sort(Fi,'descend');%%�S�x�ƦW 3 4 2 1

%% �p��k-NN two-fold cross validation

mi = [3];%%�S�x���

k = size(mi,2);
error = 0;
for fold = 1:2
    if fold == 1
        trainset = [rawData(  1: 25,mi);...
                    rawData( 51: 75,mi);...
                    rawData(101:125,mi);]; 
                  % ����C���O�e�b�A�X�֬�training set

        testset = [rawData( 26: 50,mi);...
                   rawData( 76:100,mi);...
                   rawData(126:150,mi)]; 
                  % ����C���O��b�A�X�֬�test set
    end
    
    if fold == 2
        trainset = [rawData( 26: 50,mi);...
                    rawData( 76:100,mi);...
                    rawData(126:150,mi)]; 
                  % ����C���O��b�A�X�֬�test set
          
        testset= [rawData(  1: 25,mi);...
                  rawData( 51: 75,mi);...
                  rawData(101:125,mi);]; 
                  % ����C���O�e�b�A�X�֬�training set
    end
    
[testm,~]=size(testset);
[trainm,trainn]=size(trainset);

distancev=zeros(trainm,1);%�C�Ӵ����I�P�V�m�Ϊ��ڦ��Z���q
for i=1:testm   
    for j=1:trainm
        distancev(j)=0;
        for k=1:k
            distancev(j)=distancev(j)+(testset(i,k)-trainset(j,k))^2; 
        end
        distancev(j)=sqrt(distancev(j));%.�ڦ��Z��
    end
    [val,index] = sort(distancev,'ascend');
     
    
    M = mode(label(index(1:kk)));
       
    if M ~= label(i,end)
        error=error+1;
    end  
    
end

if fold==1
CR1=1-error/testm;
fprintf('�����vCR1 = %2.4f%%\n', CR1*100)
error = 0;
end

if fold==2
CR2=1-error/testm;
fprintf('�����vCR2 = %2.4f%%\n', CR2*100)
end

end

CR = (CR1+CR2)/2;
fprintf('���������vCR = %2.4f%%\n', CR*100)






