# CS342 Supertuxkart Ice-hockey Tournament

A state based learning approach to winning a game of hockey with cars in Supertuxkart

Placed first among all state based agents

![Win]("https://github.com/Albisourous/supertuxkart-ice-hockey-writeup/blob/main/final_win.mp4")

## Group Members

Albin Shrestha, Andrew Wu, Bruce Moe, John Mackie, and Varad Thorat

## Introduction

In this project, we attempt to learn to play hockey in SuperTuxKart with a neural network using a state based agent. The goal of our AI is to be able to defeat precoded opponents and challenge other teams' AI. Our team decided to use a state based approach that learns state action relationships because the state based approach gave us access to game state information that was not provided in the image agent.
We took four different implementation approaches for our project: imitation learning, reinforcement learning, gradient-free optimization, and imitation learning with DAGGER.

## Approach Evolution

We started by implementing a basic off-policy imitation learning model on the Jurgen agent. First, we collected games where Jurgen plays itself and fed them into a network, and learned state action pairs. This initial implementation scored poorly and gave us ideas on how to continue. The first issue we encountered was making the first imitation learning model, because we had improper shuffling (sequential data) which produced very poor models. In order to fix this we did data shuffling and training on 150 games taken equally from each agent vs. jurgen on the blue team on one kart. We realized that we were only training from a single side of the field and that the network only drove in circles when we had it play the other side. So, the first main optimization was to collect action state pairs from both sides of the field.
Our next plan was to create a hand-tuned controller AI that outperformed Jurgen so that we could learn from something better than the bots we would compete with. Our initial strategy was to only steer towards the puck and move forward, setting both agents to constantly chase the puck. However, we immediately ran into a problem with this approach where our agents would eventually get stuck against a wall and be unable to reverse since we coded them to only drive forward. To fix this issue, we created an instance variable to track if an agent was stuck based on how long its velocity remained below a certain threshold, and once we determined that it was stuck we set it to reverse for a certain number of frames before proceeding to chase the puck again. While this fix solved the issue of the karts being stuck, we quickly realized that the approach of using two chaser agents was not adequate to consistently beat the TA agents or score at least one goal/game.
The next improvement we made to the naive chaser agents was to create a “goalie” agent that would chase the puck whenever the puck was between itself and the enemy goal, but if the agent ever passed the puck then it would reverse towards its own goal until it was behind the puck, and then begin chasing again. The advantage of this approach was that it would ensure that the agents would only hit the puck towards the enemy goal, rather than its own goal, and have a higher chance of having a good angle to hit the puck into the enemy goal. After creating this goalie behavior, we tried different combinations of two naive chasers, one goalie/chaser, and two goalies. We found that the goalie/chaser combination performed the best against the TA agents, but it was still not good enough to consistently beat them, especially the better agents(Jurgen). We theorize that a reason for our controller’s poor performance was due to our approach of the agent always driving towards and hitting the puck with its chasing behavior regardless of the angle, rather than first driving to a different position to get a better angle or steering to hit the puck at an angle in which it is more likely to score. However, we could not figure out how to implement this approach given our limited time and lack of experience with projective geometry.
After many hours of hand-tuning the controller, we could not get it to perform much better than Jurgen. It also acted less fluid than Jurgen and we thought this might be a harder set of actions and strategies for the imitation network to copy. So, ultimately we scrapped the idea in favor of imitating Jurgen.
Then, we decided to implement reinforcement-learning on top of our imitation model. We would first imitate Jurgen and then adapt its behavior with a reward function on top of that knowledge. We planned to reward the agent for its closeness to the puck, the puck’s velocity vector and scoring. This failed because we were unable to properly get a reward function that would teach our model properly. It would sometimes reward the bot for events where the AI's own goal. Additionally keeping track of all of the rewards was difficult to implement. We tried to modify the in-class reinforcement implementation for the SuperTuxKart racer, but we ultimately weren’t able to understand how to adapt the code to work in a more complex environment where there are many more factors and the presence of other agents can impact the effectiveness of the reward function.
After the reinforcement learning approach failed, we decided to attempt to more accurately learn to copy the Jurgen agent behavior. We did this in three ways. The first approach to optimize our reinforcement was to learn from games where Jurgen plays agents other than itself. This performed better on the grader since we previously had no examples of what Jurgen would do in situations against a bot like Yann and Yoshua. The next was to try different model structures and compare their effectiveness. We did a bit of data shuffling and training on 150 games taken equally from each agent vs. jurgen one specific one kart. With this model we got a .171 Loss with MISH and a 18/100 on the local grader. Next we added a dropout on the input layer only and reran our train. This gave us a 35/100 on the grader and a 0.281 loss. Next we just added dropouts on multiple layers which caused us to fail drastically compared to just one dropout. Then we tried a dropout on input layer only + batchnorm which gave us a 0.351 loss and 21/100 on the grader. Then we tried batchnorm on every layer giving us a 0.236 loss and 15/100 on the local grader. Surprisingly, a deeper, dense model with one dropout layer in its input structure and Mish performed better than smaller models and models with BatchNorm, more Dropouts, or different activations. We theorize this is because a denser network learns more complex functions, different features correspond to different distributions so normalizing the input is faulty, and more dropouts on a small network means we lose too much information. We also tried making better features such as velocity magnitude that should correlate to acceleration and expanding our feature space to include opponent positions, but both of these agents performed worse surprisingly. After trying many structures we finally performed as well as 76/100 (43 on Canvas) from pure imitation learning from Jurgen. Then, we decided to optimize our imitation further by adding DAGGER on top of our imitation model. We ran simulated games where our state based agent took action and we calculated the loss from what the Jurgen agent would do given the same game state. Unfortunately, our DAGGER training at first reduced the network’s score, so we reverted to the pure imitation strategy. We theorize this is due to us selectively training on the samples that the Jurgen agent won on, so once we introduced DAGGER we also taught our agent how to lose/not score goals effectively. Eventually, by training DAGGER on good games we were able to get our better, final agent.
We also tried an iterative random sampling gradient free method similar to the in class code where we sample from a distribution and perturb the weights of our model and rollout matches and choose to follow the better weights depending on the score of the matches. We played against all TA agents in each rollout with each ball position in [-1,1]. This method ended up not being able to produce a good agent in a feasible amount of epochs, so we tried seeding it with our initial DAGGER model and it ended up producing agents that were worse although some by a slim margin. This method has potential, but perhaps the “reward function” where we prune the forward passes of the model could have been more correlated with the results than just the number of goals produced by the agent and we could have chosen better features so our parameter space could be more articulate or smaller.

## Reward Function Ideas

Before implementing DAGGER, we planned on using reinforcement learning to improve our imitation model. Although we were ultimately unable to successfully train our network with Reinforced learning, due to time constraints, we had thought and partially implemented the proper reward functions we would have to improve our bot.

### Attacker + Score Car

Attacker
Gives low reward for velocity vector in direction of enemy car
Gives higher reward for long velocity vector in close proximity of enemy car
Score Car
Gives high reward for scoring
Gives high reward for ball vector that aims at enemy goal
Gives reward if the projection of the velocity of this car from the origin of the car closely aligns with the vector from the ball to the goal
Negatively rewards when the ball moves backward unless the ball is inside one of these red regions.

### Both Attackers

This strategy consists of two Score Cars that both attempt to make goals. We thought that since we are graded on average goals, winning does not matter; only making goals matters.

### Goalie + Mover

Goalie
Stays in the net for the most part
Given the opportunity where the goal and the puck are lined up this kart will full speed drive at the puck and try and snipe a goal
Gives reward for being close to the goal, but also for scoring
Mover
Gives high reward for being close to the puck
Used to move the puck around and line it up for the Goalie

## Final Implementation Review

In our final turn-in approach we use imitation learning with additional DAGGER training. First, we collect training data of the Jurgen agent playing against itself, as well as other opponents. We play against multiple opponents to make sure that we get a wide variety of state/action pairs. Then we delete all games in which the Jurgen agent does not win. If we want to beat the Jurgen agent we thought that we should learn only the actions it took when it beat itself. Then we use imitation learning to create a model that copies Jurgen. Once this model is trained we use DAGGER to improve our gameplay. We play more games, but this time we make our state agent model play the opponents. Our training loop revisits these games and calculates Jurgens actions given each state and compares them with our model's actions to generate loss values. Locally, this agent does better than the base Jurgen agent by about 10 points on the grader, scoring twice as much against Jurgen as Jurgen does to itself.

## Conclusion

In the end our model was able to score 91/100 on the test games locally and 82/100 on canvas. If we had more time we would have liked to get our reinforcement code to work and perhaps a smarter gradient-free approach. We believe that it would have been able to learn to outperform Jurgen, and therefore it would outperform many of the other students’ models who imitate Jurgen in the tournament. However, we still think our model was able to play well and hope to play well in the tournament.

## Contributing

Please contact me on LinkedIn to collaborate: https://www.linkedin.com/in/albin-shrestha
