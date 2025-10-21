#Chapter 4 -Section 4.3.1
#Finite horizon
#backwards induction to find optimal policy


na<-3
pa<-0.1  #arrival rate
ps1<-0.2 #slow service rate
ps2<-0.4
ps3<-0.6
hc<-1  #holding cost rate  

sr<-matrix(c(ps1,ps2,ps3),3,1)


#Uses index of action in reward. Reward Function appears in 41,45,46
rew<-function(ss,dd){ #lin/cubic
   1*(ss-1)    +5*(dd/5)^3
}
rew1<-function(ss,dd){  #lin/lin
  1*(ss-1)   +5*(dd/5) 
}
#Backward induction
#initialize
lambda=1
nmax<-6  #nmax number of states
d_store<-matrix( 0,1,nmax)
v_store=matrix(0,1,nmax)
dd=d_store 
loop<-1
v<-matrix(0,nmax,1)
 
m<-nmax-1
ba=matrix(0,nmax,3)
max_epoch<-10
while (loop < (max_epoch+1)) { 
#q-factors

for (s in 2:m){ 
  for (a in 1:na){ 
    ba[s,a]<-rew1(s,a) + lambda*(sr[a,1]*v[s-1] +  (1-pa-sr[a,1])*v[s] +pa*v[s+1])  
  }}

for (a in  1:na){ 
  ba[nmax,a]<-rew1(nmax,a) + lambda*(sr[a,1]*v[nmax-1] +  (1 -sr[a,1])*v[nmax]) 
  ba[1,a]<-rew1(1,a)  + lambda*((1-pa)*v[1] +  pa*v[2])
}

#Bellman equations and greedy action choice
for (s in 1:nmax){
  v[s]<-min(ba[s,])
dd[s]<-which.min(ba[s,]) 
}
  
d_store=rbind(d_store,dd)
loop<-loop+1
 }
d=d_store[-1,]
print(d)
print(v)
 
plot(0:(nmax-1),v,"l",xlab="State",ylab="Value",cex.lab=1.3)

#Need to manipulate these to get symbols in right place.
#plot for max_epoch=50  
if (max_epoch==50)  { 
drev=d[nrow(d):1,]
color=c("white","lightgrey","darkgrey")
x=seq(0.5,max_epoch+.5,1)  #note labelled backwards 
y=seq(0,nmax,1)
image(x,y,drev,col=color, xlab= "Decision Epoch", ylab="Occupancy",  cex.lab=1.3)
text( x=25, y=0.5,labels = expression(a[1]),cex=2.5)  #was 1.8
text( x=25,y=1.5,labels = expression(a[2]),cex=2.5)
text (x=25, y=4, labels = expression(a[3]),cex=2.5)
}
 
 
#plot for nmax =10 - needs adjustemnt for figures 4.1 a and b
if (max_epoch == 10) {
drev=d[nrow(d):1,]
color=c("white","lightgrey","darkgrey")
x=seq(0.5,max_epoch+.5,1)  #note labelled backwards 
y=seq(0,nmax,1)
image(x,y,drev,col=color, xlab= "Decision Epoch", ylab="Occupancy",  cex.lab=1.3)
text( x=7.5, y=4,labels = expression(a[1]),cex=2.5)  #was 1.8
#text( x=7.5,y=4,labels = expression(a[2]),cex=2.5)
text (x=2, y=4, labels = expression(a[3]),cex=2.5)
}
 
