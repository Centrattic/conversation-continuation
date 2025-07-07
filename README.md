# conversation-continuation
Continue conversations / simulate people better than purely in-context methods (give to Claude, ask to continue conversation)

Goal: simulate conversations between you/other person. End up with model that can do either depending on whether you start prompt with "Riya: hi whats up" or like "{Friend name}: hi whats up." But maybe a single friend simulator just works better? Cleaner data or something? Idk. Easiest for SFT to start with conversation simulator I think, switch to friend simulator later.

## Ideas
1. some sort of diffusion option -- too hard to get good I think? Not trying to train a general-purpose LLM
2. PEFT -- empirically seems like would be pretty good with right regularization! Different LoRAs for different people.
3. Baseline: just try in-context only

## Notes

Ok. We're going with PEFT to start. Things to consider?
* context window size: like how many messages back in the conversation history should I include for SFT? Probably going to make this tunable, because greatly depends on the conversation. I should definitely chunk by time to not break this (ie. context should not include messages from hours ago unless I want to predict when next person is gonna talk, and I don't want continuous simulator yet)
* tokenizer is super important here. I should have a token for the person who's speaking (either Riya or Friend) at start of each turn. Maybe I should finetune the tokenizer too?
* i'm not planning to include time in the prompt right now - don't know if I have enough data to get good at this?
* what is architecture for the PEFT? Probably LoRA, I want attention still or something, though tbh even just very generalized bigram learning would be interesting (but getting it to be so general requires more than 1 attention layer)
* Adding special tokens actually appears to be a problem, since not seen tokens -- connectes with other rare tokens, so we get much more OOD outputs. So we will keep [RIYA] and [FRIEND] format, just not use special tokens (add them as new). Only good to add new if I am finetuning very long... Also, adding [RIYA] as special token but not [FRIEND] means model really only outputs [FRIEND] response (this is before much training btw) since [RIYA] token is now very rare and not been seen before -- interesting behavior, maybe useful for making the single-person simulator, but then doesn't actually use the [RIYA] responses...

ex. training sample
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

### Running list of bugs to figure out/things to do
* ```completion = decoded[len(prompt_text)-2:].strip() # why -2 idk`` Why do I need -2 here?
* Why, on different training runs, are the same prompts selected by random.choice for sample generation callback? Does random choice go through the same order or smt?
* Figure out how to handle creating new log file, but have a flag for continuing trianing or something if you cancel it that allows you to append to old logs. For now, assume no continuation.
* Save outputs form inference_compare to file for nice view - maybe view in dashboard or smt

## Future Ideas

So I somewhat fundamentally doubt this idea, because it relies on the assumption that predicting next token in conversation is equivalent to predicting the nuances of how someone acts. But the state space of next thing the person will talk about is huge, and requires like deeper understanding of someone's values? Also, there's a reason that character AI is still pretty bad/models are not great at simulating real people -- good character training requires some RL I think (?) and like reflection on values to ensure consistency in some way, and models will not be as consistent as humans without this.

Ok, despite this, future ideas:
* should check how my tokenizer is with emojis, ideally have a tokenizer with emojis/model trained on data including these
* should consider whether or not to simulate one person instead of two (I think this is worse right now, because don't have person outside of this conversation history)