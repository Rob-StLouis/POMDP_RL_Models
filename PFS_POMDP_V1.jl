
using Distributions
using StatsBase
using Gadfly

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




#takes the number of states, the n for each state, and randomly assigns groups to new categories
function CRP(state_array:: Array{Int64,2},  n_obs::Int64 ,n_states::Int64,α::Float64,TransMatrix::Array{Int64,2})
    if n_states==0
        return 1
    end
    weights = float(state_array[1,1:(n_states+1)])
    div = float(n_obs+α)
    weights[n_states+1] = α
    weights = weights ./ div

    weights = reshape(weights,(n_states+1)) .*  TransMatrix[1:n_states+1]

    weights = weights ./ sum(weights)

    return rand(Categorical(weights))
end




(1+.001 )*.999

##likelihood function
#takes the new state assigned to the array, and gives it a bigger function.
# so observations will be a one dimensional vector integer values, indicating their category

function P_Weights(obs:: Array{Int64,1}, cues_array:: Array{Float64,4},reward_array:: Array{Float64,4})
    P = (reward_array[obs[1]]+1) / (sum(reward_array[1:2])+1)
    #the last three areas indicate the underlying variables.
    P = P * pdf(Beta(.9,.9),(P+.1) *.9 )

    for i in 2:length(obs)
        #I think we need the I -1 here
        P = P * ((cues_array[1,1,i-1,obs[i]] + 1) / (sum(cues_array[1,1,i-1,1:4]+1)))
    end
    return P
end


#
function ResampleW(weight_vec)
    warray = WeightVec(weight_vec)
    new_index = sample(collect(1:length(weight_vec)),warray,length(weight_vec),replace=true)

    return new_index
end

# Gets the observed data, and weights it.
function ProbWeightVec(obs:: Array{Int64,1}, cues_array:: Array{Float64,4})
    R = 1
    #the second dimension indicates reward
    for i in 2:length(obs)
        R = R * ((cues_array[1,1,i-1,obs[i]] + 1) / (sum(cues_array[1,1,i-1,1:4]+1)))
    end
    return R
end


function ProbReward(Reward_array:: Array{Float64,4})
    #just returns the probability of reward.

    R = (Reward_array[1,1,1,2] + 1) / (sum(Reward_array[1,1,1,1:2]+1))
    return R
end


## weighting function

function gen_weights(CuesArray::Array{Float64,4},NumStates::Int64,StateCount::Array{Int64,2},
    new_observed::Array{Int64,1},
    n_obs::Int64,alpha::Float64,
    TransMatrix::Array{Int64,2})
        #generates new state
    new_state = CRP(StateCount, n_obs, NumStates,alpha,TransMatrix)
        if new_state > NumStates
            P_num_states = NumStates +1
        else
            P_num_states = NumStates
        end

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



(1:20)/10

##simulating  the response
nparticles = 5000


First_Second= zeros(Float64,20)
    Second_Third = zeros(Float64,20)
    ProbFirst = zeros(Float64,20)
    ProbSecond = zeros(Float64,20)

maxstates=30
transMatrix = ones(Int64,maxstates,maxstates,nparticles)



eyI = eye(maxstates)*60
for i in 1:nparticles
        transMatrix[1:maxstates,1:maxstates,i] = transMatrix[1:maxstates,1:maxstates,i]+ eyI
end

for R1 in 1:20

a = zeros(Float64,nparticles,100,3,4)
numstates = zeros(Int64,nparticles)
state_count = zeros(Int64,nparticles,100)
same_state = zeros(Int64,nparticles,3)

weights = Array(Float64,nparticles)
    laststate = ones(Int64,nparticles)
probResponse3 = Array(Float64,110)


probWvec = Array(Float64,nparticles)
predprodvec = Array(Float64,nparticles)
postPredCheck3 = Array(Float64,110)
nindex = 0
stim_index = [0,0,0]
nindexrec = Array(Float64,110,nparticles)
s_round = 0
lasttrial1=15
lasttrial2=40
lasttrial3=lasttrial2+1
@time for i in 1:lasttrial3
    if i < lasttrial1+1
        s_round = 1
        elseif i < lasttrial2-1
        s_round = 2
    else
        s_round = 3

    end

    stim_index[1] = rand(Bernoulli(states[s_round].p_reward))+1
    stim_index[2] = rand(Categorical(states[s_round].p_cue1))+1
    stim_index[3] = rand(Categorical(states[s_round].p_cue2))+1
    print("i:")
    print(i)
    print(" ")

    for k in 1:nparticles

        new_state, weights[k] , probWvec[k] , predprodvec[k], numstates[k] =  gen_weights(
        a[k,1:100,1:3,1:4],
        numstates[k],
        state_count[k,1:100],
        stim_index,
        i,
            (R1/.1),
            transMatrix[laststate[k],1:maxstates,k])

        state_count[k,new_state] = state_count[k,new_state] + 1
        a[k,new_state,1,stim_index[1]] = a[k,new_state,1,stim_index[1]] +1
        for j in 2:3
            a[k,new_state,j,stim_index[j]] = a[k,new_state,j,stim_index[j]] + 1
        end
        if i == lasttrial1
            same_state[k,1] = new_state
        end

        if i == lasttrial2
            same_state[k,2] = new_state
        end

        if i == lasttrial3
            same_state[k,3] = new_state
        end
            laststate[k] = new_state


    end
    weight2 = weights ./ sum(weights)
    probWvec = (probWvec ./ sum(probWvec))
    probResponse3[i] = sum(probWvec .* predprodvec)
    postPredCheck3[i] = sum(weight2 .* predprodvec)
    nindex = ResampleW(weights)
    nindexrec[i,1:nparticles] = nindex
    a = a[nindex,1:100,1:3,1:4]
    if i == lasttrial1
        same_state[1:nparticles,1] = same_state[nindex,1]
    end

    if i == lasttrial2
        same_state[1:nparticles,2] = same_state[nindex,2]
    end

    if i == lasttrial3
        same_state[1:nparticles,3] = same_state[nindex,3]
    end



    numstates = numstates[nindex]
    state_count = state_count[nindex,1:100]
end

    First_Second[R1] =  mean(same_state[1:nparticles,1] .== same_state[1:nparticles,2])
    Second_Third[R1] =  mean(same_state[1:nparticles,1] .== same_state[1:nparticles,3])
    ProbFirst[R1] =  probResponse3[lasttrial1]
    ProbSecond[R1] =  probResponse3[lasttrial2]
end



plot(x=First_Second,y=Second_Third, Geom.line)

plot(x=numstates,Geom.histogram)
plot(x=same_state[1:nparticles,1],y= same_state[1:nparticles,2],Geom.point)

Second_Third

transMatrix[2,1:maxstates,3]

plot(x=1:80,y=probResponse3)


plot(x=1:110,y=probResponse3,Geom.line, Scale.y_continuous(minvalue=0,maxvalue=1))

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
