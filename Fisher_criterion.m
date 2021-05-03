clc;       % 清除command window
clear      % 清除workspace
close all  % 關閉所有figure

%% 讀取.txt資料
dataSet = load('iris.txt');
rawData = dataSet(:,1:4);    % 原始資料，75筆資料 x 4個特徵
label   = dataSet(:,5);      % 75筆資料所對應的標籤
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
[Firank,index] = sort(Fi,'descend');%%特徵排名 3 4 2 1

%% 計算k-NN two-fold cross validation

mi = [3];%%特徵選取

k = size(mi,2);
error = 0;
for fold = 1:2
    if fold == 1
        trainset = [rawData(  1: 25,mi);...
                    rawData( 51: 75,mi);...
                    rawData(101:125,mi);]; 
                  % 選取每類別前半，合併為training set

        testset = [rawData( 26: 50,mi);...
                   rawData( 76:100,mi);...
                   rawData(126:150,mi)]; 
                  % 選取每類別後半，合併為test set
    end
    
    if fold == 2
        trainset = [rawData( 26: 50,mi);...
                    rawData( 76:100,mi);...
                    rawData(126:150,mi)]; 
                  % 選取每類別後半，合併為test set
          
        testset= [rawData(  1: 25,mi);...
                  rawData( 51: 75,mi);...
                  rawData(101:125,mi);]; 
                  % 選取每類別前半，合併為training set
    end
    
[testm,~]=size(testset);
[trainm,trainn]=size(trainset);

distancev=zeros(trainm,1);%每個測試點與訓練及的歐式距離量
for i=1:testm   
    for j=1:trainm
        distancev(j)=0;
        for k=1:k
            distancev(j)=distancev(j)+(testset(i,k)-trainset(j,k))^2; 
        end
        distancev(j)=sqrt(distancev(j));%.歐式距離
    end
    [val,index] = sort(distancev,'ascend');
     
    
    M = mode(label(index(1:kk)));
       
    if M ~= label(i,end)
        error=error+1;
    end  
    
end

if fold==1
CR1=1-error/testm;
fprintf('分類率CR1 = %2.4f%%\n', CR1*100)
error = 0;
end

if fold==2
CR2=1-error/testm;
fprintf('分類率CR2 = %2.4f%%\n', CR2*100)
end

end

CR = (CR1+CR2)/2;
fprintf('平均分類率CR = %2.4f%%\n', CR*100)






