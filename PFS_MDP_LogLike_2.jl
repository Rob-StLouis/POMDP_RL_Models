
using Distributions
using StatsBase
using Gadfly
using DataFrames

# ended up being more efficent to use separate arrays rather than new complex types.
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




# logistic transform
x =3
y = exp(x)/(1+exp(x))
print(y)

logistic(x)

#takes the number of states, the n for each state, and randomly assigns groups to new categories
function CRP(state_array:: Array{Int64,2},  n_obs::Int64 ,n_states::Int64,lastState::Int64,TransMatrix::Array{Int64,2},α::Float64 = 2.0)
    if n_states==0
        return 1
    end

    #ordinary CRP
    weights = float(state_array[1,1:(n_states+1)])
    div = float(n_obs+α)
    weights[n_states+1] = α
    weights = weights ./ div


    #This is autoregressive process
    stabilityVec = TransMatrix[lastState,1:n_states+1][1:n_states+1]

    weights = weights .* transpose(stabilityVec)
#     weights = stabilityVec

    weights = weights ./ sum(weights)
    weights = reshape(weights,n_states+1)

    return rand(Categorical(weights))
end




#create new probability of reward, for both objects
#we are going to create a new
function LogOddDrift(reward_array::Float64)
        reward_array = reward_array + rand(Normal())
    return reward_array
end



pdf(Bernoulli(.3),0)

#gets the likelihood of observations, including reward, for each particle
function P_Weights(obs:: Array{Int64,1}, cues_array:: Array{Float64,4},reward_array:: Float64)


    P = pdf(Bernoulli(logistic(reward_array)),obs[1])


    for i in 2:length(obs)
        P = P * ((cues_array[1,1,i-1,obs[i]] + 1) / (sum(cues_array[1,1,i-1,1:4]+1)))
    end
    return P
end


# resamples the particles
function ResampleW(weight_vec)
    warray = WeightVec(weight_vec)
    new_index = sample(collect(1:length(weight_vec)),warray,length(weight_vec),replace=true)

    return new_index
end

# Gets the observed data, and weights each subsequent particle
function ProbWeightVec(obs:: Array{Int64,1}, cues_array:: Array{Float64,4})
    R = 1
    #the second dimension indicates reward
    for i in 2:length(obs)
        R = R * ((cues_array[1,1,i-1,obs[i]] + 1) / (sum(cues_array[1,1,i-1,1:4]+1)))
    end
    return R
end


#probability of reward for each particle.
function ProbReward(Reward_array:: Float64)
    #just returns the probability of reward.

    R = logistic(Reward_array)
#     print(" RewardProb ")
    return R
end


x =rand(Normal(0,.2),1,3)
x[1]

##overall controller function, generates the weight for the particle given the just the cues, the probability of reward given the cues,
#and the likelihood

function gen_weights(CuesArray::Array{Float64,4},pRewardArray::Array{Float64,2}, NumStates::Int64,StateCount::Array{Int64,2},
    new_observed::Array{Int64,1},
    TransMatrix::Array{Int64,2},lastState1::Int64, n_obs::Int64)
        #generates new state

    new_state = CRP(StateCount, n_obs, NumStates, lastState1,TransMatrix)

    parray = pRewardArray

    #updates reward

        if new_state > NumStates
            P_num_states = NumStates +1
#         print(" addstate ")
        else
            P_num_states = NumStates
#         print(" keepstate ")
        end
#     print(new_state)

    newlogprob = pRewardArray[1,1:P_num_states] + rand(Normal(0,.00000000001),1,P_num_states)

    parray[1,1:P_num_states]= newlogprob


     nlp = newlogprob[new_state]
        #get the probability of a new point.
    #this is likelihood of all observations, including reward
    W = P_Weights(new_observed,CuesArray[1,new_state,2:3,1:4],nlp)

    #likelihood for just the cues
    PW = ProbWeightVec(new_observed,CuesArray[1,new_state,2:3,1:4])
    #predicted probability of reward
    PredP = ProbReward(nlp)
        #update the observed counts
    return new_state, W, PW, PredP, P_num_states, parray
end

nstates = 3
states = Array(M.LatentStates,nstates)

#low salience condition, only 2 cues used in this vector

states[1] = M.LatentStates([1,0,0,0],[1,0,0,0],0)
states[2] = M.LatentStates([1,0,0,0],[1,0,0,0],1)
states[3] = M.LatentStates([0,1,0,0],[1,0,0,0],0)





nstates=10

nparticles = 10000

nrounds = 24

a = zeros(Float64,nparticles,nstates,3,4)



pvec = rand(Normal(0,30000),nparticles,nstates)
numstates = zeros(Int64,nparticles)
state_count = zeros(Int64,nparticles,nstates)

transMatrix = ones(Int64,nstates,nstates,nparticles)


laststate = ones(Int64,nparticles)
numstatevec = zeros(Int64,nparticles)


weights = Array(Float64,nparticles)
probResponse = Array(Float64,nrounds)

probWvec = Array(Float64,nparticles)
predprodvec = Array(Float64,nparticles)
postPredCheck = Array(Float64,nrounds)

nindex = 0
stim_index = [0,0,0]
nindexrec = Array(Float64,nstates,nparticles)
s_round = 0

#relects empirical results

@time for i in 1:nrounds

    if i < 4
         s_round = 1

        elseif i == 6

        s_round = 1

        elseif i == 11

        s_round = 1

        elseif i == 15

        s_round = 1

        elseif i == 18

        s_round = 1

        elseif i == 20

        s_round = 1
    else
        s_round = 2
    end




    stim_index[1] = rand(Bernoulli(states[s_round].p_reward))
    print(stim_index[1])
    stim_index[2] = rand(Categorical(states[s_round].p_cue1))+1
    stim_index[3] = rand(Categorical(states[s_round].p_cue2))+1
    print("i:")
    print(i)
    print(" ")

    for k in 1:nparticles

        numInit = numstates[k]

        new_state, weights[k] , probWvec[k] , predprodvec[k],numstates[k], pvec[k,1:nstates] =  gen_weights(
        a[k,1:nstates,1:3,1:4],
        pvec[k,1:nstates],
        numstates[k],
        state_count[k,1:nstates],
        stim_index,
        transMatrix[1:nstates,1:nstates,k],
        laststate[k],
        i)

        state_count[k,new_state] = state_count[k,new_state] + 1
        a[k,new_state,1,1] = a[k,new_state,1,1] +1
        for j in 2:3
            a[k,new_state,j,stim_index[j]] = a[k,new_state,j,stim_index[j]] + 1
        end

        laststate[k] = new_state



    end
    weight2 = weights ./ sum(weights)
    probWvec = (probWvec ./ sum(probWvec))
    probResponse[i] = sum(probWvec .* predprodvec)
    postPredCheck[i] = sum(weight2 .* predprodvec)
    nindex = ResampleW(weights)
    a = a[nindex,1:nstates,1:3,1:4]
    numstates = numstates[nindex]
    laststate = laststate[nindex]
    pvec = pvec[nindex,1:nstates]
    state_count = state_count[nindex,1:nstates]
    transMatrix = transMatrix[1:nstates,1:nstates,nindex]


end
