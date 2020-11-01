function  R = dist(a,b)
if length(a)~=length(b)
    error('please enter two values with same length !');
end
R = 0;
 for i = 1:length(a)
     R = R+(a(i)-b(i))^2;
 end
 R = sqrt(R);
end