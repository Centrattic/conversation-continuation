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

### Version 2: I'm adding TDA! (7/18/25)
Ideas:
* So first idea here is obviously influence functions
* But wondering if there's another way easily to like do TDA! Oh!! One thought is I just take the activation vector on the current output (across the sequence) or last token or whatever (ideally I average across sequence) and then compare this to cached activation vectors (averaged across sequence again) for various input data samples for the final model! Then I just choose topk as most heavily attributed to 
* Ok the second idea seems pretty intuitive and easy. Since this model isn't on TransformerLens (also eventually I want to do other activation based interp) and TransformerLens is also reallly bulky, could use BauKit easily or nnsight. Nnsight seems pretty good (also someone on OSMI made nninterp recently, could look at this?)
* Ok but influence functions are the main method in field rn, so start with that. Can also try this second TDA idea tomorrow. 
* influence function implementation following: https://www.lesswrong.com/posts/sYeZvofqbWJDrXEHM/influence-functions-why-what-and-how
* another thing to look into is attribution patching: https://www.neelnanda.io/mechanistic-interpretability/attribution-patching, but maybe this simple cosine similarity activation idea is starting to look good 😭
* Ok influence fucntions seem to give you influence of a given data point on the output logits -- ah, maybe that would be cool but harder to get back attributions from that without a lot of brunt work. Okay but it only needs to happen once. Implementing these would be really cool and fun :D I have been missing math a bit
* Ok nvm influence functions are super expensive (backward pass atelast). Let's start simple with the activation extraction. Literally just requires one forward pass


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