function op=Evaluation_measures(mgt,mev)

%The function to find out the evaluation measures
%mgt is ground truth mev is evaluated result
% mgt and mev is N by 4 matrix 
% first column X coordinate
% second column Y coordinate
% third column is width of rect
% four column is the hight

%re assigning the size
[c d]=size(mgt);
[e f]=size(mev);
op=[];


%mapping array formation gt to mv
ma1=zeros(c,e);
tia=0;
for i=1:c
for j=1:e
    t1=mgt(i,:);
    k1=mev(j,:);
    
block1=[t1(1) t1(2) t1(3) t1(4)];
block2=[k1(1) k1(2) k1(3) k1(4)];
interArea=rectint(block1,block2);
if (interArea>0.25*min([t1(3)*t1(4) k1(3)*k1(4)]))
    ma1(i,j)=1;
end
%For calculating total inter area    
tia=tia+rectint(block1,block2);
end
end

ma2=ma1';
%%%%%%%%%+++++++++++=======================================================
if (c>1&&e>1)

%Number of false detection
if c==1
    FalseDet=sum((ma1)==0);
elseif c==0
    FalseDet=e;
else
FalseDet=sum(sum(ma1)==0);

end
%Missed regions
if e==1
    Misreg=sum((ma2)==0);
elseif e==0
    Misreg=c;
else
Misreg=sum(sum(ma2)==0);

end
%find no of under segment
k=sum(ma1)-1;
%nomber of under segment components usc
h=k(find(k>0));
[w1 usc]=size(h);
us=sum(k(find(k>0)));
%find the no of over segment component osc
k=sum(ma2)-1;
h=k(find(k>0));
[w2 osc]=size(h);
os=sum(k(find(k>0)));

%find the area under the ground truth
gta=0;
for j=1:c
    t1=mgt(j,:);
    gta=gta+t1(3)*t1(4);
    
    
end

%find the area under the ground truth
eva=0;
for j=1:e
    t1=mev(j,:);
    eva=eva+t1(3)*t1(4);
    
    
end

% Find nomber of correctly segmented regions
a1=find(sum(ma1)==1);
a2=find(sum(ma2)==1);
[s1 x1]=size(a1);
[s2 x2]=size(a2);
cs=0;
for i=1:x1
    for j=1:x2
        if ma1(a2(j),a1(i))==1
            cs=cs+1;
        end
    end
end


% tia/gta

%out put the calculated error features
op=[FalseDet,Misreg,us,os,gta,eva,tia,cs,osc,usc];
%%%%%%%++++++==============================================================
elseif (c==1&&e>1)
    FalseDet=Count(ma1,'==0');
    if (sum(ma1)==0)
        Misreg=1;
    else
        Misreg=0;
    end
    us=0;
    os=Count(ma1,'==1')-1;
    gta=mgt(3)*mgt(4);
    eva=0;
for j=1:e
    t1=mev(j,:);
    eva=eva+t1(3)*t1(4); 
end
    cs=0;
    if os>0
    osc=1;
    else osc=0;
    end
    usc=0;
    op=[FalseDet,Misreg,us,os,gta,eva,tia,cs,osc,usc];

elseif (e==1&&c>1)
    if (sum(ma2)==0)
       FalseDet=1; 
    else FalseDet=0;
    end
    Misreg=Count(ma1,'==0');
    us=Count(ma1,'==1')-1;
    os=0;
       eva=mev(3)*mev(4);
    gta=0;
for j=1:e
    t1=mgt(j,:);
    gta=gta+t1(3)*t1(4); 
end
    cs=0;
    osc=0;
    if us>0
        usc=1;
    else usc=0;
    end
    
op=[FalseDet,Misreg,us,os,gta,eva,tia,cs,osc,usc];
elseif (c==1&&e==1)
    if (ma1==1)
      FalseDet=0;
      Misreg=0;
      us=0;
      os=0;
      cs=1;
      osc=0;
      usc=0;
    elseif (ma1==0)
      FalseDet=1;
      Misreg=1;
      us=0;
      os=0;
      cs=0;
      osc=0;
      usc=0;
    end
    gta=mgt(3)*mgt(4);
    eva=mev(3)*mev(4);
    op=[FalseDet,Misreg,us,os,gta,eva,tia,cs,osc,usc];
elseif(e<1)
    FalseDet=0;
      Misreg=c;
      us=0;
      os=0;
      cs=0;
      osc=0;
      usc=0;
      gta=0;
for j=1:c
    t1=mgt(j,:);
    gta=gta+t1(3)*t1(4); 
end
    eva=0;
    op=[FalseDet,Misreg,us,os,gta,eva,tia,cs,osc,usc];
    %total inter area (tia), nomber of under segment components usc ,
%     no of over segment component osc, area under the ground truth eva
elseif (c==0)
    op=[1 1 1 1 1 1 1 1 1 1];
end


end