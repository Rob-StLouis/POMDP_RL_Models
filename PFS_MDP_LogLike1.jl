
using Distributions
using StatsBase
using Gadfly
using DataFrames




module M
type Particle
    weight:: Float64
    #goes, state, cue type, cue value
    Cues::Array{Float64,3}
    #state, cue value
    ProbR::Array{Float64,2}
    StateCount::Array{Float64,1}
    num_states:: Int64
end


Particle() = Particle(1.0,zeros(Int64,100,2,4),zeros(Int64,100,2),zeros(Int64,100),0)

type LatentStates
    p_cue1:: Array{Float64,1}
    p_cue2:: Array{Float64,1}
    p_reward:: Float64
end

LatentStates() = LatentStates([.25,.25,.25,.25],[.25,.25,.25,.25],0.0)

end 
    
using M




vector(3)

#takes the number of states, the n for each state, and randomly assigns groups to new categories
function CRP(state_array:: Array{Int64,2},  n_obs::Int64 ,n_states::Int64,
    lastState::Int64,TransMatrix::Array{Int64,2},
    eraseState::Bool,
    α::Float64 = .5)
    
    
    if n_states==0
        return 1
    end
    weights = float(state_array[1,1:(n_states+1)])
    div = float(n_obs+α)
    weights[n_states+1] = α
    weights = weights ./ div
    
    
    #removes the bias
    if eraseState == false
        stabilityVec = TransMatrix[lastState,1:n_states+1][1:n_states+1]
        weights = weights .* transpose(stabilityVec)

    else
#         weights = sqrt(sqrt(weights))
    
        weights = ones(length(weights))
        
    end
        
        


    

    
    weights = weights ./ sum(weights)
    weights = reshape(weights,n_states+1)
    
#     print(" v2:")
#     print(weights)

#     weights = ones(nstates+1) ./ (nstates+1)
    return rand(Categorical(weights))
end




function MDP(state_array:: Array{Int64,2},  n_obs::Int64 ,n_states::Int64,
    lastState::Int64,TransMatrix::Array{Int64,2},
    eraseState::Bool,
    α::Float64 = .5)
    
    
    if n_states==0
        return 1
    end

    #removes the bias
    if eraseState == false
        weights = float(TransMatrix[lastState,1:n_states+1][1:n_states+1])
        

    else
#         weights = sqrt(sqrt(weights))
        weights = ones(n_states+1)
        
    end
    
# #     print("nstates: ")
#     print(typeof(weights))
#     print(weights)
    
    weights[n_states+1] = α
    
    
    
    
    
    
    weights = weights ./ sum(weights)
    weights = reshape(weights,n_states+1)
    
#     print(" v2:")
#     print(weights)

#     weights = ones(nstates+1) ./ (nstates+1)
    return rand(Categorical(weights))
end

weights = TransMatrix[lastState,1:n_states+1][1:n_states+1]

weights[3] =5
weights


TransMatrix = Array(Int64,100,100,1)
lastState=3
n_states=10
stabilityVec = TransMatrix[lastState,1:n_states][1:10]


transpose([.5,.2])

sqrt(10)

##likelihood function
#takes the new state assigned to the array, and gives it a bigger function.
# so observations will be a one dimensional vector integer values, indicating their category

function P_Weights(obs:: Array{Int64,1}, cues_array:: Array{Float64,4},reward_array:: Array{Float64,4})
    P = (reward_array[obs[1]]+1) / (sum(reward_array[1:2])+1)
    #the last three areas indicate the underlying variables.
#     P = P * pdf(Beta(.9,.9),(P+.1) *.9 )
    
    for i in 2:length(obs)
        #I think we need the I -1 here
        P = P * ((cues_array[1,1,i-1,obs[i]] + 1) / (sum(cues_array[1,1,i-1,1:4]+1)))
    end
#     print(" P_weights ")
    return P
end


#
function ResampleW(weight_vec)
    warray = WeightVec(weight_vec)
    new_index = sample(collect(1:length(weight_vec)),warray,length(weight_vec),replace=true)
    
#     print(" Resample ")
    return new_index
end 

# Gets the observed data, and weights it.
function ProbWeightVec(obs:: Array{Int64,1}, cues_array:: Array{Float64,4})
    R = 1
    #the second dimension indicates reward
    for i in 2:length(obs)
        R = R * ((cues_array[1,1,i-1,obs[i]] + 1) / (sum(cues_array[1,1,i-1,1:4]+1)))
    end
#     print(" ProbWVec ")
    return R
end


function ProbReward(Reward_array:: Array{Float64,4})
    #just returns the probability of reward. 
    
    R = (Reward_array[1,1,1,2] + 1) / (sum(Reward_array[1,1,1,1:2]+1))
#     print(" RewardProb ")
    return R
end


## weighting function

function gen_weights(CuesArray::Array{Float64,4},NumStates::Int64,StateCount::Array{Int64,2},
    new_observed::Array{Int64,1},TransMatrix::Array{Int64,2},
    lastState1::Int64,
    n_obs::Int64,
    eraseState1::Bool,
    alpha::Float64)
        #generates new state
    new_state = MDP(StateCount, n_obs, NumStates, lastState1,TransMatrix,eraseState1,alpha)
        if new_state > NumStates
            P_num_states = NumStates +1
#         print(" addstate ")
        else 
            P_num_states = NumStates
#         print(" keepstate ")
        end
#     print(new_state)
    
        #get the probability of a new point.
        #this is where it conditions on the new particle's state.
        W = P_Weights(new_observed,CuesArray[1,new_state,2:3,1:4], CuesArray[1,new_state,1,1:2]) 
        #add the state info to the new points
    #     addd this later 
        #probability weights for reward vector
        PW = ProbWeightVec(new_observed,CuesArray[1,new_state,2:3,1:4])
        #comment
        PredP = ProbReward(CuesArray[1,new_state,1,1:2])
        #update the observed counts
    return new_state, W, PW, PredP, P_num_states
end

nstates = 3
states = Array(M.LatentStates,nstates)

states[1] = M.LatentStates([1,0,0,0],[1,0,0,0],0)
states[2] = M.LatentStates([1,0,0,0],[1,0,0,0],1)
states[3] = M.LatentStates([1,0,0,0],[0,1,0,0],0)


states[2]

##simulating  the response
nparticles = 5000
# a =Array(M.Particle,nparticles)
# for i in 1:nparticles
#     a[i] = M.Particle()
# end


#the array configuration
#particle, state, dimension, cue,instance

    First_Second2= zeros(Float64,20)
    Second_Third2 = zeros(Float64,20)
    ProbFirst = zeros(Float64,20)
    ProbSecond = zeros(Float64,20)
    ProbThird = zeros(Float64,20)

a = zeros(Float64,nparticles,maxstates,3,4)
numstates = zeros(Int64,nparticles)
state_count = zeros(Int64,nparticles,maxstates)
same_state = zeros(Int64,nparticles,3)

lasttrial1=15
lasttrial2=40
lasttrial3=lasttrial2+1

ntrial=lasttrial3

maxstates = 30


transMatrix = ones(Int64,maxstates,maxstates,nparticles) 
    
#insane bias 
    eyI = eye(maxstates)*80
for i in 1:nparticles
        transMatrix[1:maxstates,1:maxstates,i] = transMatrix[1:maxstates,1:maxstates,i]+ eyI
end


for R1 in 1:20
    
    
    

    



# ntrial = 45

a = zeros(Float64,nparticles,maxstates,3,4)
numstates = zeros(Int64,nparticles)
state_count = zeros(Int64,nparticles,maxstates)
same_state = zeros(Int64,nparticles,3)



laststate = zeros(Int64,nparticles)
numstatevec = zeros(Int64,nparticles)


weights = Array(Float64,nparticles)
probResponse = Array(Float64,ntrial)

probWvec = Array(Float64,nparticles)
predprodvec = Array(Float64,nparticles)
postPredCheck = Array(Float64,ntrial)
nindex = 0 
stim_index = [0,0,0]
nindexrec = Array(Float64,100,nparticles)
s_round = 0
    

    @time for i in 1:ntrial
    if i < lasttrial1+1
        s_round = 1
        elseif i < lasttrial2
        s_round = 2
    else
        s_round = 3
        
    end
    
    #delated reward
    
#     if i < 4
#          s_round = 1
        
#         elseif i == 4
        
#         s_round = 1
        
#         elseif i == 6
        
#         s_round = 1
        
#         elseif i == 9
        
#         s_round = 1
        
#         elseif i == 13
        
#         s_round = 1
        
#         elseif i == 18
        
#         s_round = 1
#     else
#         s_round = 2
#     end
        
        
        
        

    stim_index[1] = rand(Bernoulli(states[s_round].p_reward))+1  
    stim_index[2] = rand(Categorical(states[s_round].p_cue1))+1
    stim_index[3] = rand(Categorical(states[s_round].p_cue2))+1  
#     print(stim_index)
    print("i:")
    print(i)
    print(" ")
    
#         if i == 2
#             estate = true
#             elseif i == 7
#             estate = true
#         else
            estate = false
#         end
    
    for k in 1:nparticles
#         print(" k:")
#         print(k)
        numInit = numstates[k]
        

        
        #         toggles whether or not you are using the pseudo-mdp
#         estate=true
        
        new_state, weights[k] , probWvec[k] , predprodvec[k], numstates[k] =  gen_weights(
        a[k,1:maxstates,1:3,1:4],
        numstates[k],
        state_count[k,1:maxstates],
        stim_index,
        transMatrix[1:maxstates,1:maxstates,k],
        laststate[k],
        i,
        estate,
            (R1/40))
#         print("ran")
        
#         print("new_state: ")
#         print(new_state)
        state_count[k,new_state] = state_count[k,new_state] + 1
        a[k,new_state,1,stim_index[1]] = a[k,new_state,1,stim_index[1]] +1
        for j in 2:3
            a[k,new_state,j,stim_index[j]] = a[k,new_state,j,stim_index[j]] + 1
        end 
#         if numstates[k]>numInit
#             transMatrix[numstates[k],numstates[k],k] = 2
#         end
        
        
#         #keeps it from failing when it starts with zero states.
#         if numInit>0
#                     transMatrix[laststate[k],new_state,k] = transMatrix[laststate[k],new_state,k]+1 

#         end
            
        

    
            
#         print(new_state)
        laststate[k] = new_state
#             print(lasttrial3)
            
        if i == lasttrial1
            same_state[k,1] = new_state
        elseif i == lasttrial3
#                 print("XXXXXXXX")
#                 print(new_state)
            same_state[k,3] = new_state
                elseif i == lasttrial2
            same_state[k,2] = new_state
        end
            

        
        

    
    
        
    end
    
        
    weight2 = weights ./ sum(weights)
    probWvec = (probWvec ./ sum(probWvec))
    probResponse[i] = sum(probWvec .* predprodvec)
    postPredCheck[i] = sum(weight2 .* predprodvec)
    nindex = ResampleW(weights)
#     nindexrec[i,1:nparticles] = nindex
    a = a[nindex,1:maxstates,1:3,1:4]
    numstates = numstates[nindex]
    laststate = laststate[nindex]
    state_count = state_count[nindex,1:maxstates]
    transMatrix = transMatrix[1:maxstates,1:maxstates,nindex]
        
   if i == lasttrial1
        same_state[1:nparticles,1] = same_state[nindex,1] 
    end
    
    if i == lasttrial2
        same_state[1:nparticles,2] = same_state[nindex,2] 
    end
        
    if i == lasttrial3
        same_state[1:nparticles,3] = same_state[nindex,3] 
    end
        
#         print(same_state[1:10,1:2])
    
end
    First_Second1[R1] =  mean(same_state[1:nparticles,1] .== same_state[1:nparticles,2])
    Second_Third1[R1] =  mean(same_state[1:nparticles,1] .== same_state[1:nparticles,3])
    print(mean(same_state[1:nparticles,1] .== same_state[1:nparticles,3]))
    ProbFirst[R1] =  probResponse[lasttrial1]
    ProbSecond[R1] =  probResponse[lasttrial2]
    ProbThird[R1] =  probResponse[lasttrial3]


end

(1:10/40

h = plot(x=probWvec,y = predprodvec,Scale.x_continuous(minvalue=0,maxvalue=.0005))


plot(x=First_Second1,y=Second_Third1, Geom.line)

plot(x=ProbThird,y=Second_Third1-First_Second1, Geom.line)

ProbSecond

h = plot(x=numstates,Geom.histogram,Scale.x_continuous(minvalue=0,maxvalue=5))
# draw(PNG("GradualExtinction.png",6cm,6cm),h)

plot(x=1:ntrial,y=probResponse[1:ntrial],Geom.line, Scale.y_continuous(minvalue=0,maxvalue=1))

p = plot(x=40:80,y=probResponse[40:80],Geom.line,
Guide.xlabel("Trial"), Guide.ylabel(""), Guide.title("Unfamiliar CS"), Scale.y_continuous(minvalue=0,maxvalue=1) )
draw(PNG("FamiliarLowNoTrans.png",6cm,6cm),p)
writetable("FamiliarLowNoTrans.csv",DataFrame(Trial = 1:80,Response=probResponse))

plot(x=40:80,y=probResponse[40:80],Geom.line,
Guide.xlabel("Trial"), Guide.ylabel(""), Guide.title("Familiar CS"))

plot(x=1:60,y=probResponse3)


n = Array(Float64,10000000)
w = Array(Float64,10000000)
k=0
for i in 1:nparticles
    for j in 1:numstates[i]
        k=k+1
        a1 = a[i,j,1,1]
        a2 = a[i,j,1,2] 
        n[k]= a2 / (a1+a2) 
        w[k]= a1+a2
    end
end
        
        

n=n[1:k]

plot(x=n,Geom.histogram)

w = WeightVec(w[1:k])

x = sample(n,w,100000,replace=true)

plot(x=x,Geom.histogram(bincount=20))

function takesq(x)
    y = x^2
    z = x^3
    return z, y, "red"
end

a1 = a[1,1,1,1]
a2 = a[1,1,1,2]
a1/(a1+a2)

a[1,1,1,1]



rand(Categorical([.5,.5]))

pdf(Beta(.5,.5),.0001)

#so the issue is that you don't have any way of identifying which state you are in,until you get to the end, which is reall

 + eye(15)

for i in 1:5
    state = i 
end

state

function P_Weights(InitialLocation)
    pdf
