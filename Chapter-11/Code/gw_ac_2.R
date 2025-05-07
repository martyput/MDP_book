#this updates critic after updating actor

create_grid_world <- function(rows,cols) {
  grid_world <- matrix(0, nrow = rows, ncol = cols)
   grid_world[rows,cols] <- 1  # Goal state
  #grid_world[4,4] <- 1  
  return(grid_world)
}

# Define the possible actions
actions <- c("up", "down", "left", "right")

# Define the step function
step <- function(state, action, grid_world) {
  row <- state[1]
  col <- state[2]
  
  if (action == "up") {
    row <- max(1, row - 1)
  } else if (action == "down") {
    row <- min(rows, row + 1)
  } else if (action == "left") {
    col <- max(1, col - 1)
  } else if (action == "right") {
    col <- min(cols, col + 1)
  }
  next_state <- c(row, col)
  reward <- ifelse(grid_world[row, col] == 1, 10, -0.1)
  done <- grid_world[row, col] == 1

  return(list(next_state = next_state, reward = reward, done = done))
}

#don't use this
grid_to_linear <- function(row, col, n_rows, n_cols) {
  return((row - 1) * n_cols + col)
}

softmax <- function(x,b=10  ) {
  exp_x <- exp(x/b)   
  return(exp_x / sum(exp_x))
}

# Function to select an action using softmax probabilities based on policy weights
select_action <- function(row,col,policy_weights) {
  action_probabilities <- softmax(policy_weights[row,col, ])
  action_index <- sample(actions, 1, prob = action_probabilities)
  return(action_index)
}
 
gen_episode=function(grid_world,policy_weights,eps,iter_max=500) {
    accum_rew=numeric()
    accum_state=numeric()
    accum_action=numeric(0)
   state <- c(1, 1)  # Start at top-left corner
  # state=c(sample( 1:(rows-1),1),sample(1:(cols-1),1))
    
    done <- FALSE
    iters=1
    
    while (!done) {
       #use softmax to select actions and eps-greedy
       action1=select_action(state[1],state[2],policy_weights)
       action2=sample(c("up","down","left","right"),1)
       action=ifelse( runif(1)<eps ,action2,action1)
      result <- step(state, action, grid_world)
      next_state <- result$next_state
    #cat(state,action,reward,"\n")
      done <- result$done
      accum_rew=c(accum_rew,result$reward)
      accum_action=c(accum_action,action)
    #note states are pairs of row,col
      accum_state=c(accum_state,state)
      state <- next_state
      iters=iters+1
      if (iters>iter_max) {done= TRUE}
      }
    return(list(accum_rew,accum_action,accum_state)) }
 
update_critic <-  function(ep_results) {
  #ep_results is output from gen_episode
  #this creates setup for backward loop
  rewards=ep_results[[1]]
  tot_rewards <- numeric(length(rewards))
  cumulative_reward <- 0
  for (t in seq(length(rewards), 1, by = -1)) {
    cumulative_reward <- rewards[t] +   cumulative_reward
    tot_rewards[t] <- cumulative_reward  #this contains sums necessary for policy gradient
  }
  states=matrix(ep_results[[3]],length(rewards),2,byrow=TRUE) 
  
  #evaluate critic with least squares
  row=matrix(states[,1],length(rewards),1)
  col= matrix(states[,2],length(rewards),1)
  reg_lin=lm(tot_rewards~row+col  +I(row^2) +I(col^2))
  values=reg_lin$fitted
  reg_coeff =reg_lin$coefficients
  return(list(coef=reg_coeff,tot_val=tot_rewards[1],values=values))
}
   

update_weights<- function(ep_results,reg_coef,values,lr) {
  #ep_results is output from gen_episode
  #this creates setup for backward loop
  #this is calculated in crit but it is recacluated herw
  
  rewards=ep_results[[1]]
  tot_rewards <- numeric(length(rewards))
  cumulative_reward <- 0
  for (t in seq(length(rewards), 1, by = -1)) {
    cumulative_reward <- rewards[t] +   cumulative_reward
    tot_rewards[t] <- cumulative_reward  #this contains sums necessary for policy gradient
  }
 states=matrix(ep_results[[3]],length(rewards),2,byrow=TRUE) 
 #evaluate critic with least squares
 row=matrix(states[,1],length(rewards),1)
 col= matrix(states[,2],length(rewards),1)

#values are critic values
  

#Now update weights - assume first indicators
   
  for (t in 1:length(states[,1])) {
    state <-  states[t,1:2]
    action <- ep_results[[2]][t]
    #cant use action name to refer to column
   act_idx=(action=="up")*1 +(action=="down")*2+(action=="left")*3+(action=="right")*4
    Gt <-  tot_rewards[t]
    
    # Compute the softmax probabilities
   # action_probabilities <- softmax(policy_weights[state[1],state[2], ])
    # Compute the gradient
    #gradient <- -action_probabilities
    #gradient[act_idx] <- gradient[act_idx] + 1
     #print(gradient)
    gradient=comp_gradient(policy_weights,state,action_idx)
    # Update the policy weights
   #use critic as baseline
    row=state[1]
    col=state[2]
    vec=c(1,row,col,row^2,col^2)
    baseline = sum(reg_coef*vec)
    print(baseline)
    #This is where critic is used
    #baseline=values[t]
    policy_weights[state[1],state[2], ] <-  policy_weights[state[1],state[2], ] + lr * (Gt-baseline) * gradient
  }
  #cat("policy weights",policy_weights,"\n")
  return(policy_weights)
} 

comp_gradient<-function(policy_weights,state,action_idx)
{action_probabilities <- softmax(policy_weights[state[1],state[2], ])
 gradient <- -action_probabilities
gradient[act_idx] <- gradient[act_idx] + 1
  return(gradient)
  }

# Train the agent
set.seed(43)
num_episodes <-5000
 
ww=numeric()
rows=10
cols=10
policy_weights=array(rnorm(rows*cols*4),dim=c(rows,cols,length(actions)))
gw=create_grid_world(rows,cols) 
reg_coef=c(8,.1,.1,0,0)

for (n in 1:num_episodes){
  eps=20/(100+n)
ep_results=gen_episode(gw,policy_weights,eps)
lr=40/(400+n)
policy_weights=update_weights(ep_results,reg_coef,crit$values,lr)
crit=update_critic(ep_results)
reg_coef=crit$coef
ww=c(ww,crit$tot_val)
print(c("n=",n))
print(policy_weights)
}
 
 
gw 
for (row in 1:rows) {
  for (col in 1:cols) {
    action_probabilities <- softmax(policy_weights[row,col, ])
    cat("row","col",actions,"\n")
    cat(row,col,action_probabilities,"\n")
  }
}

plot(1:1000,ww[1:1000],"l",cex=.1,xlab="iteration",ylab="value",cex.lab=1.2)
