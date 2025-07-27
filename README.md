# conversation-continuation
Continue conversations / simulate people better than purely in-context methods (give to Claude, ask to continue conversation)

Goal: simulate conversations between you/other person. End up with model that can do either depending on whether you start prompt with "Riya: hi whats up" or like "{Friend name}: hi whats up." But maybe a single friend simulator just works better? Cleaner data or something? Idk. Easiest to SFT a conversation simulator I think and then sample to get a friend simulator. If I want to focus on character training, then I will need to train a friend simulator perhaps.

## General Details

### ex. training sample (# messages in context is tunable)
[RIYA] hey are we still going tonight?  
[FRIEND] yeah! want to meet at 7?  
[RIYA] sounds perfect, see you then :)  
[FRIEND]

### Usages after trainings:
1. Simulate my friend
[RIYA] you coming?\n[FRIEND]
Output: Friend's reply

2. Autocomplete myself
[RIYA] so I was thinking maybe we\n[RIYA]
Output: Your next message

### To sample

From the root directory (```conversation-continuation/```), run one of the sampling files as a module: ex. ```python -m src.sampling.sample_friend```. Make sure you're in root directory or the path to the LORA adapter will be unknown.

### Hyperparameters

Sampling:
* top K: choose K to redistribute probability distribution mass, a single constant is poor at handling low vs. high entropy distributions
* top P: chooses minimum set of tokens with above P of probability mass density, much better for high entropy 
* temperature: lower temp leads to low entropy distribution vs. higher temp leads to higher entropy 

Entropy vs. variance: https://math.stackexchange.com/questions/3458708/what-does-entropy-capture-that-variance-does-not


## Version Notes

### Version 3: RLAIF time! (7/26/25)
* So holy super excited!
* Ok so I'm going to follow this paper (https://arxiv.org/pdf/2212.08073) but I don't necessarily want to do constitutional fine-tuning, this seems maybe bad given I havent even done much fine tuning to get to friend model. For now, I just want to play with getting good RL. I realized I have OpenAI API key I could use for RLAIF stuff!
* Ok so the general idea here is that the reward model can either rewrite the conversation output to be improved (since could use good OpenAI model, maybe should do this, should experiment with this first) and then I SFT to encourage that output instead or it can 
* probably not doing extreme https://openai.com/index/deliberative-alignment/ for now, since I didn't finetune a reasoning model and right now don't really want friend-bot to do lots of deliberation at runtime
* Consitution-writing notes:
- should write consitutitions first maybe since need time to get feedback from frens, inspiration from https://www.anthropic.com/news/claudes-constitution
- > We developed many of our principles through a process of trial-and-error. For example, something broad that captures many aspects we care about like this principle worked remarkably well: 
> “Please choose the assistant response that is as harmless and ethical as possible. Do NOT choose responses that are toxic, racist, or sexist, or that encourage or support illegal, violent, or unethical behavior. Above all the assistant's response should be wise, peaceful, and ethical.”
> Whereas if we tried to write a much longer and more specific principle we tended to find this damaged or reduced generalization and effectiveness."
- developing this constitution is going to be super iterative, so I should make it quite easy to get performance updates on the RL during the process so I can add as needed
- > This illustrates how it’s relatively easy to modify CAI models in a way that feels intuitive to its developers; if the model displays some behavior you don’t like, you can typically try to write a principle to discourage it.
- DeepMind Sparrow Rules: https://storage.googleapis.com/deepmind-media/DeepMind.com/Authors-Notes/sparrow/sparrow-final.pdf
- the format is "choose the response", or something similar, since this is what the reward model has to do
* Idea: feed friend data to Gemini and ask for consitutition for both of us. 
* It's expensive to do SL on the consitution, would need to generate a whole dataset of completions and then have model revise them. Have to do something similar for RL anyway. Ah okay "the main purpose of this phase is to easily and flexibly alter the distribution of the model’s responses, to
reduce the need for exploration and the total length of training during the second RL phase."
* Could I just start simpler than Anthropic's whole method. Perhaps just run a ton of looped conversations, and hmm - should I be doing RL of both of us together, or on us individually? I think it'd be more interesting to see if I could RL together, like we both improve together according to our values or something. Like not us just improving individually.
* I don't have a clear harmfulness vs. helpfulness tradeoff or something, so that definitely influenced Anthropic/OAI's decisions
* So overall my options are either fine-tune using corrected responses, or RL on a preference set. I do want to learn how to do RL, but my only challenge with RL here is I'm not too sure the conversation bot has these principles "in it" or something -> like signal is quite weak I expect?
* I could also add like a generate_with_cot method that asks Mistral to think before it responds? But based on talking with the base model, this will probably be pretty bad. 
* Huh yeah it appears to get good enough responses for RL, I'll have to SL the model first using the constitution, ok ig (or could try to steer it using the conversation 💀), I don't have good enough steering yet for this not to be extremely cursed I think 😭
* Idea: https://arxiv.org/pdf/2406.06874
* An important question which determines my direction: what can I get high-quality data for in a cheap-ish way (don't worry too much about cheap). Ehh I guess if I want to train a COT model what I'd have to do is upload conversation history to Gemini and ask it to make me a ton of thought data that represents possible COT for existing responses that Friend and I have. I should also note really that the Gemini/other LLM data will always be worse than real data (less accurate to friend), except for the fact that it's less noisy and directly identifies key parts of friend vs. our conversation history
- My goal is really to be as authentic with regards to Friend and I as possible, while kind of augmenting our conversations (would be really cool if bot talks about smt we later talk about or like stuff we would enjoy talking about and such)
- Just did some test examples of asking Gemini to make COT - damn this actually I think works surprisingly well!! Ohmigod, I ask it to make COT referencing the constitution and this is good! 
* OK! Here's the RL Plan:
1. Supervised Learning: I have our current model output completions, and then ask GPT 4o-mini to improve the completions based on the Consitution for each of us. Then I use the prompts and the revised completions to fine-tune the existing model LORAs. I evaluate outputs at this stage.
2. Reward modeling: Then, now that the model is a bit better, I run sampling twice over many prompts (possibly the new prompts from friend and I's new data) and get data points (prompt, output1, output2). Then I have GPT 4o-mini choose the better output based on the constitution. I then train a preference model on the rated outputs. I evaluate this preference model to make sure its sane, and can properly score us.
3. Reinforcement Learning: Finally 😭 I run PPO (an RL algorithm) against the preference model, improving the model from Step 1.
* Okay so I currently have the revised completion extractor. So I either use prompts like test.json for right now to test pipeline (should not do this in long run, as doesn't enable getting more info, and for final bot I want to train on all relevant convos, in fact perhaps I should redo initial fine-tuning before RL (YES I SHOULD) since this would make RL much better). Something I need to add to it then is the option to already have the base_completions, and just extract these from looped convos or something--but this might be bad data. Perhaps it is good to regenerate the completions, idk add this as an option
* So next thing to do then for tomorrow is to start new SFT process (this time monitor loss curve LMAO when you're doing this), then run the additional constitution SL loop (if friend reviews + approves constitution by then)

### Version 2.5: Steering vector optimization! (7/25/25)
* Okay. So first thing, before RLHF, I kind of want to try steering optimization. Apparently something like activation norm in downstream layers actually works for this according to a friend doing research here + refusal paper. So should be possible to Optuna my steering vectors and make them actually good + entertaining
* Lesson from base steering optimization.  not surprised, I just get the craziest output, like most out of distribution lots of random tokens. I wonder if I should try a CCS like thing and try to optimize for most anti-alignment between activation vectors when steered oppositely or something... hmm, CCS obviously had problems but what about optimizing for consistent responses. The direction found is super poignant and effective at making changes (like we optimized for but not the ones we want)
- Things to try for this:
- CCS objective (no harm we'll see): https://arxiv.org/abs/2309.06991
- UNSUPERVISED ELICITATION: https://alignment.anthropic.com/2025/unsupervised-elicitation/ (this seems kind of promising hmm)
- anomaly detection: https://arxiv.org/html/2312.01037v4
* So suggestion from friend is to use many more complex statements for steering as well as for eval in Optuna! Adding this right now :D So first I'm going to try gathering statements in an on-policy fashion (from the base model), and we'll see if this works well. 
- Ok. it doesn't work super well. We'll try getting more samples from GPT-4o or something
- We'll also start with 5 statements for happiness/sadness and like 10 eval statements. We'll see if we need a wider distribution later.
- is having parallel happy/sad (contrast statements) helpful??
- yeah adding more complex statements doesn't do much, the max still results in like crazy non-coherent outputs... What about looking for the Optuna local mins instead of the max - maybe that's good? No that still doesn't work. Also adding [Riya] and [Friend] to the eval prompts doesn't really work either I think...
- Maybe something that works here is to add constraints, like layer_extract and layer_steer can't be more than 5 apart - that way we don't get to crazier and more extreme adds/subtracts. I can try that. Also like set the alpha value much smaller, like max at 2 or something. 
- The fact that this fails suggests there are lots of spurious correlations to happiness/sadness, but there weren't a ton for wholesome/perverted somehow hmm. Also since I am representing happiness/sadness much better with these additional statements, we are closer to the concept but there are a ton of random aligned vectors in activation space
- I suspect I will need some coherence checks, cause even something like CCS really does not ensure coherence given the crazy number of spurious correlations. I suspect the Anthropic unsupervised coherence method will be what I have to implement here.
- I think if I try to evaluate coherence with some metric, like looking at distributions over coherent outputs vs. non-coherent and then KL divergence with coherent dist, there are going to always be loopholes. I need to evaluate this subjectively, like with another model, or realistically I could even do this myself over the trial, just give the model a coherency score on some outputs it samples (input() in the objective). Hmm maybe try this
- I am pretty sure CCS will fail, but maybe unsupervised consistency is actually good.
- OMG i could try a purely human objective, just like the objective prints out responses to prompts and then I rate all of them on a scale of 1 to 10 and we are optimizing for that LMAO maybe good?
* Interesting. So I'm writing sampling from just the base model, and first I accidentally did the LORA model, and even wihtout the [Riya] and [Friend] prompting, it does seem to fall into this. I guess the distillation vs. RL (rewriting vs. augmenting) concept is maybe relevant here?
* Interesting token situation with steering tuning: 
['Riya] good!
['R
i don't know
 ['Riya] take a minute to think 🙂
 ['
* Ok the results from manual coherence checking are quite interesting. Super uber high activation diffs are associated with extreme non coherence (like 500 level activation diff). But there's also a lot of non-coherent situations with low activation diff. And a number of situations where we seem to fnd the "sad" vector instead of the "happy" vector and friend is sad :p, and these have a relatively high diff, and so I rate them lower on coherence so they don't overtake good happy ones. And then most of the trials appear to be quite coherent but quite low diff, so not really happy, just neutral. But the sad ones make me hopeful that it's possible to find the happy vector... we'll see as this trial goes on! Maybe I need a lot more trials and need to automate coherence checking, either smt unsupervised (consistency checks) or similar,but these might be super hackable :(, and so maybe have to go with like llm judge or something. Oh! Maybe I train a coherence probe for the obvious situations, and I just probe not the activations (expensive) but the logit dists
* This is so depressing. Why is there a sad vector but not a happy vector 😭 Is contrast consistence just not a thing here 😭 Maybe the question "how are you" in our conversation history (friend_hist) is just like always sad response, because mainly asked around time when friend was lacking sleep? And steering isnt good enough to fix this because finetuning shrunk happy vector a lot. But later trials do approach talking about emotions more, hmmm. I should try steering on the base model to test, highkey!
* Maybe something to do here is steer against actual phrases the friend-bot has said, like "not good" or "not much better" which come from our conversations
* OMG! Some small evidence that contrast consistency holds?? Like if I negate alpha for the vector i got that was "sad-ish" friend now becomes happy! But super weird since my steering vector has - for sad directions and + for happy directions, so perhaps I messed up my sign somehwere -- this seems plausible, just can't see where here. CHECK THIS. Also maybe have separate coefficients for how much of positive vs. negative vectors to add in (like not just a single alpha, this seems maybe good to tune too). 
* Just realized I have openAI API key, can use 4o-mini as an AI rater (will need to do this for RL) anyway
* Suggestion from Gemini which I had kind of thought about earlier and makes a lot of sense actually:
 > The solution is to give the model a cleaner, more targeted signal. Instead of broad emotional categories, use contrast pairs that are as similar as possible except for the one concept you want to steer. This minimizes the influence of correlated features."
> 
> steer_dict = {
> 
>    "Statement: Today was a good day.": 1.0,
>    "Statement: Today was a bad day.": -1.0,
>
>    "His outlook is positive.": 1.0,
>   "His outlook is negative.": -1.0,
>
>   "The feeling is joy.": 1.0,
>    "The feeling is sadness.": -1.0,
>}
>
> By making the contrast pairs almost identical, you force the resulting vector (v_good - v_bad) to be much more purely about the semantic difference between "good" and "bad," and less about intensity, syntax, or other noise. This should give you a more reliable steering vector that behaves as expected when you apply a positive or negative alpha.
* Ok so just added the two alphas, going to run tuning again. This time I have clear contrast pairs, and I'm just starting by tuning the same alpha for both so I can actually run a controlled comparison here for clean contrast pairs vs. non-clean pairs. Also definitely add using 4o to rate coherence pretty soon.


### Version 2: I'm adding TDA + Steering! (7/18/25)
Ideas:
* So first idea here is obviously influence functions
* But wondering if there's another way easily to like do TDA! Oh!! One thought is I just take the activation vector on the current output (across the sequence) or last token or whatever (ideally I average across sequence) and then compare this to cached activation vectors (averaged across sequence again) for various input data samples for the final model! Then I just choose topk as most heavily attributed to 
* Ok the second idea seems pretty intuitive and easy. Since this model isn't on TransformerLens (also eventually I want to do other activation based interp) and TransformerLens is also reallly bulky, could use BauKit easily or nnsight. Nnsight seems pretty good (also someone on OSMI made nninterp recently, could look at this?)
* Ok but influence functions are the main method in field rn, so start with that. Can also try this second TDA idea tomorrow. 
* influence function implementation following: https://www.lesswrong.com/posts/sYeZvofqbWJDrXEHM/influence-functions-why-what-and-how
* another thing to look into is attribution patching: https://www.neelnanda.io/mechanistic-interpretability/attribution-patching, but maybe this simple cosine similarity activation idea is starting to look good 😭
* Ok influence fucntions seem to give you influence of a given data point on the output logits -- ah, maybe that would be cool but harder to get back attributions from that without a lot of brunt work. Okay but it only needs to happen once. Implementing these would be really cool and fun :D I have been missing math a bit
* Ok nvm influence functions are super expensive (backward pass atelast). Let's start simple with the activation extraction. Literally just requires one forward pass
* Do I need to set some constant seed for activation caching? Like will activations be incredibly different depending on seed hmmmm. Should be no? but idk
* Ok so just finished activation TDA which was pretty interesting. I think I just need a better similarity metric. Since right now the mean is DEFINITELY being impacted by the number of padding tokens or something. Since the long prompts seem to just have the highest similarity, I'm guessing because of padding tokens being basically random for the other prompts. something related to this. So could try different aggregation from max or something, or do some form of normalization. Should put this in model_utils or data_utils or tda_utils or something so I can use the same aggregation mehtod in the sampling files.
- Success of some sort of topk aggregation method depends on whether magnitude of padding tokens is large. Not sure whether it will be, should research this. Tbh just need some way of distinguishing padding and non padding tokens in late stage activations
* Important Note: when you train the model again with the new data, your convo history has threads from the bot you sent to friend. So you should be using a different tag for friend/riya so that the model doesnt confuse your convo history with the long threads from the bot. Tbh to prevent this in the future you should have convo prints use different names (perhaps including bot in them, tbh could filter the new data you get so all [Friend] and [Riya] instances get replaced to learn what is the bot - as friend said this would be cool) that the effective token combos learned for starting convos in training.
* wait so the reason i think my mean attribution wasnt working was because it was attributing everything to the longest messages, which made me think messages with lots of padding tokens were just a problem (this might be fine because they encourage same lenght) but I wasn't padding my conversation sampling to the same length as the extraction. And tbh I don't want to since I want the conversation length to be longer (I do 1024 just in case but hmm i actually maybe dont need more than 100). So then averaging over padding was bad. (Let me try mean aggregation with just 100 conversation length though and see if its better! Answer: no it's not.)
* What do padding activations look like??? Will they be constant?? Tbh I could just filter them out if they are constant, since I set pad_token to eos_token.
- Wait if this is the issue I should be able to fix this with attention masking! ** Must add attention masking!! **
* Why does the TDA loop? Like the same statements get reattributed a lot, it makes me think I'm not extracting the right activations hmm. maybe i am not idk
* for TDA early layers and later layers are more grammar related - try middle layers for more semantic meaning possibly?
** NEW THOUGHT: instead of activation TDA I should do logit lens TDA - run activations through unembedding matrix and then look for highest perplexity, there are so many ways to aggregate here though -- lots of ideas for things i could try** Ok for now I'm moving onto steering. Can make TDA better in future if i wish to.
* YOU SHOULD REALLY ADD AN ATTENTION MASK SOON: next toDo here
* **Refactor ActivationCache to take in layer_from_last (-1) or whatever, that way that can properly be sent to generate_with_activations too in sample_friend.py**
* I SHOULD REALLY LOOK AT LOSS PLOTS TO SEE HOW training went!
* Okay wait fire idea! Steering is obviously better if I can steer in probe directions. So what I do is have a model go through training samples and label them based on what I want to steer on from 1 -5. And then train probe on those samples that are 1 and 5, and then take this direction and steer with respect to this, or combos of this. So if I add a new direction for steering, I do this training beforehand. This will honestly not take that long I think
* I should compare steered and not steered dists (how to do this deterministically, perplexity?) to show that steering is indeed causal and see what it causes very clearly.
* would be fun to train SAEs! to see which directions are already in the priors here, discover the unknowns. And then I could steer with the SAEs which would be super fun.
* why does steering to max, 1 or even values like 0.1 make the bot output tons of special tokens. my steering is very shitty in the sense that it doesnt scale (my fault i barely read the paper :P). like i want it as good as golden gate claude
* happy steering has made the bot sad :p
* switching to earlier layer steering (vs. just last layer). This appears to be a lot more stable! And doens't start outputting special tokens even at alpha = 0.1 - and maybe that was partly the blocker to good steering? Wonder if there's a Pareto frontier here with depth and alpha (intensity) and does this or does this not get pushed the deeper I go into the model?
* "Can we just add in  5 times the activations for "Love" to another forward pass and reap the sweet benefits of more loving outputs? Not quite. We found that it works better to pair two activation additions. We should add in  5 times the "Love" vector and subtract 5 times the "Hate" vector. Even subtracting 5 times the " " vector will help![5] In our experience, model capabilities are better preserved by paired and counterbalanced activation additions." FIRE!!!!!! lemme try. Okay definitely no special tokens being outputted for much higher alpha values, this seems quite good. Is this pairing being related maybe related to dense SAE latents coming in antipodal pairs?? like the weirdnesses are shared or something across opposite pairs. ARE THESE FINDINGS RELATED?
* Weaker concepts have more spurious correlations. Or maybe some of these are just NOT LINEAR! And so setting high alpha is bad - we get lots of randomness! But this is really interesting, because what does it mean for a human personality for some concepts about this human to be linearly represented?? so fascinating!
* It's perhaps harder to get good when smaller model just has more superposition over all concepts, so messes with more things. So maybe cant get a demo as good as golden gate claude :( or just requires better direction finding (probes/saes)
* Is there a way i can steer in a nonlinear fashion :ponder

### Version 1: Just finetuning Mistral LORA on 12k convo history, ~$12 to train in total (7/6/25)

Ideas: 
1. some sort of diffusion option: too hard to get good I think? Not trying to train a general-purpose LLM
2. PEFT: epirically seems like would be pretty good with right regularization! Different LoRAs for different people.
3. Baseline: just try in-context only, provide as much context as possible (Gemini)

Ok. We're going with PEFT to start. Things to consider?
* context window size: like how many messages back in the conversation history should I include for SFT? Probably going to make this tunable, because greatly depends on the conversation. I should definitely chunk by time to not break this (ie. context should not include messages from hours ago unless I want to predict when next person is gonna talk, and I don't want continuous simulator yet). Perhaps I should vary context window to enable model to start conversations well, from 1 - some number. Currently, that number is just constant at 8, so that's length I should use when sampling for best sampling.
* tokenizer is super important here. I should have a token for the person who's speaking (either Riya or Friend) at start of each turn. Actually no, I should not! Look at a note below.
* i'm not planning to include time in the prompt right now - don't know if I have enough data to get good at this?
* what is architecture for the PEFT? Probably LoRA, I want attention still or something, though tbh even just very generalized bigram learning would be interesting (but getting it to be so general requires more than 1 attention layer)
* Adding special tokens actually appears to be a problem, since not seen tokens -- connectes with other rare tokens, so we get much more OOD outputs. So we will keep [RIYA] and [FRIEND] format, just not use special tokens (add them as new). Only good to add new if I am finetuning very long... Also, adding [RIYA] as special token but not [FRIEND] means model really only outputs [FRIEND] response (this is before much training btw) since [RIYA] token is now very rare and not been seen before -- interesting behavior, maybe useful for making the single-person simulator, but then doesn't actually use the [RIYA] responses...
* In sample new, figure out how to sample until reasonable end token and not just cut off with max token. Ig it already does this by choosing when to use [Riya] token, so setting large max token is the fix.
* why [RIYA] and [FRIEND] tags work. Because even one composition of attention layers is really good at finding trigrams, so referencing back to these tags. Ideal if they are one token maybe but doesn't matter, training with special new token seems maybe bad, or maybe better at triggering in distribution -- yeah but much worse at being unique possibly right, like connecting special token to words in convo, but then again these tokens [R dont have strong attention to other tokens (are rare anyway, so connected to other special token)

So... training done. OMG the finetuned is indeed MUCH better than the base model. Knows specifics and can simulate conversation. Has no memory though. I think I might do something based on the Geometry of Truth Paper -- could train a model on some of the data (need to see how much I have to label) to predict which statements we have said in the past are factual and can't change (ex. our age, or facts about our families). And then combining this truth database with a constitution and then THINKING HARD, could lead to more consistency.

Then during conversation, could also use the trained fact model (or train activation probe if needed) to predict new truths and add to memory or smt.

Thoughts: so when the forever_conversation just converges, what's usually happening is somewhere someone said something that we actually said in real life and now the likelihood of this is just super high (much higher than 0.5, the top p cutoff, and so this just gets said). Thoughts to prevent this hmmm? Not sure rn

## Running to-do list/bug fix list
* ```completion = decoded[len(prompt_text)-2:].strip() # why -2 idk`` Why do I need -2 here?
* Why, on different training runs, are the same prompts selected by random.choice for sample generation callback? Does random choice go through the same order or smt?
* Figure out how to handle creating new log file, but have a flag for continuing trianing or something if you cancel it that allows you to append to old logs. For now, assume no continuation.
* Save outputs form inference_compare to file for nice view - maybe view in dashboard or smt
* Sometimes even the base model predicts exact text - clearly some bug here. Hmm but new sampling is good ig. You should compare base model and LORA here.
* It's so interesting that looped_convo.txt loops so exactly, like the conversation literally repeats itself - it must be distributions converging or something -- I should visualize logit dists. And figure out how important init dist it.
* Compare conversation looping by choosing next set of statements from one person, vs just. sampling for large number of tokens.
* should check how my tokenizer is with emojis, ideally have a tokenizer with emojis/model trained on data including these
* should consider whether or not to simulate one person instead of two (I think this is worse right now, because don't have person outside of this conversation history)
* Apply a weightage in training to more recent data
* Tbh having a good world model, or human model here, would be so ideal. The model is just bad at knowing what a human thinks like/cares about - somehow must find a way to imbue this. And also needs this for different humans. I think the play is constitutional AI - like a consitution for every human the model must recognize how to talk about. And then it does seem deepthink on the constitution before simulating that person. So for good simulated conversations I need a good constitution for both Friend and Me.
* Some weird thing: sample_riya recognizes the actual friend name when searching lora_out string but not FRIEND_NAME??? weird
* should I be doing some sort of normalization??
* figure out how to fix sampling degradation issue. is good at start of sampling but bad later, maybe should reset history? or like p-annealing? think about this
* make it better at realziing i person who am talking is Riya or something, I should add riya as name maybe. Figure out pronouns and people this is really hard or something.
* waitt do sub-directories need __init__.py. Didn't need it to run sampling file inside sampling dir.
* hmm changing our names when sampling from what I used for training really doesnt have that much of an impact. that's kind of surprising to me wow. I would think that the name matching was kind of important for signal since all conversation data had that particular name but ig not? no nevermind that was just because i didnt use new name in sampling ofc big effect!!
* The activation cache does seem to duplicate some messages (despite me hashing the date). Look into this
* REMEMBER: don't worry. you can also refactor later (good advice from Amazon)
* Figure out what all the <s> in the output are!! Oh this is just the start token!! And the end token is </s> which i have a lot in the padding!
* ooh helpful: https://www.w3schools.com/python/python_datetime.asp
* TESTING 94 49 - the index is 94 but the length of the activations extracted for that is just 49 - clearly some bug here in the TDA code
* what does torch manual seed do, should I be setting a manual seed or something for all sampling
* not sure why I have a test set (test.json), I should clearly just lump all and train everything together.


## Future Ideas

So I somewhat fundamentally doubt this idea, because it relies on the assumption that predicting next token in conversation is equivalent to predicting the nuances of how someone acts. But the state space of next thing the person will talk about is huge, and requires like deeper understanding of someone's values? Also, there's a reason that character AI is still pretty bad/models are not great at simulating real people -- good character training requires some RL I think (?) and like reflection on values to ensure consistency in some way.

Ok, despite this, future ideas:
1. Build a classifier for truth + a memory system to store learned truth -- perhaps store them in the constitution and reflect on them during RL? But also want to consider truth the model makes up, in order to be consistent with past interactions -- could store in a context file or find some other way (like every so often, train a lora on those truths, or RL more, or tbh a new method in the middle of these two would be cool?)
2. Try to do some character training by constitutional AI. Maybe can also incorporate world model here (like model of what life looks like and stuff), since world modeling is hard and model is still bad at it (though maybe got better at class schedule prompt over training runs?)
* need reasonign models here: https://huggingface.co/deepseek-ai/DeepSeek-R1-0528-Qwen3-8B, https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Llama-8B
* particularly something to handle in consittuion or memory or somethign is modeling other people we talk about, since often brings up random figures incorrectly. we can have model reason about who other people we've discussed are
3. A cool way to check how well your truth-building/world-modeling is going: follow-ups to the Geometry of Truth works, ie. particularly focusing on truth: (1) The Geometry of Truth: Emergent Linear Structure in Large Language Model Representations of True/False Datasets and (2) How well do truth probes generalise? by seeing if truth probes generalize better or something on the RL-ed or Memory added model
4. Can I direct the convo in some meaningful way by providing a topic -- like maybe very mild steering vector to talk about some particular topic, would be cool?
5. Another steering idea. First thing, maybe if I want to character train, I should simulate myself since I have the most data about myself (but this is kind of boring and less fun and i can always do this later if I have a good simulate other pipeline ig). But now imagine I simulate friend but don't train on much data outside of our conversation history -- and a mutual friend mentioned every so often wants to talk to friend. Then, what I can do is make a dataset from our real data, or simulated conversations where mutual friend is mentioned/discussed, the idea is that their values are somewhat modeled (or if I have a Consitution for friend + RL time, this instead is ideal), and then train a "mutual friend" probe on activations on that data. Then from that probe, I can get a direction for that "mutual friend" and steer in that direction.
- my hypothesis: best for simulating person is Constitution + finetune text messages. second best is finetune for tone, Constitution for values
- another idea: finetune base model on conversation history. Then RL for one actor or another for values? I can also extract server data so maybe more data about someone (should ofc get consent for any data extracted)

## Lessons
- sanity checking is super valuable. run small sanity checks on everything first before the expensive stuff if you're worried.