import pygame
import random
from model import DQNetwork, Memory
import numpy as np


#pygame.mixer.pre_init(44100,16,2,4096)
pygame.init()

score = 0
health = 100
loop = 1
difficulty = 1

iter1 = 0
iter2 = 0

display_width = 800
display_height = 600
contra_height = 165
contra_width = 108
enemy_height = 80

gameDisplay = pygame.display.set_mode((display_width,display_height))
pygame.display.set_caption('Contra Revamped')
clock = pygame.time.Clock()

contraImg = pygame.image.load('contra.png')
enemyImg = pygame.image.load('enemy.png')

music = pygame.mixer.music.load('background.mp3')
pygame.mixer.music.play(-1)

#shotCollide = pygame.mixer.Sound('blast.mp3')

font = pygame.font.SysFont('arial',60)

x = 0
y = int(display_height*0.35)

x_shot = contra_width
y_shot = y + 58
x_travel = x_shot

y_change = 0

shotsX = []
shotsY = []
enemyX = []
enemyY = []

gameExit = False

while not gameExit:

	for event in pygame.event.get():
		if event.type == pygame.QUIT:
			gameExit = True

		if event.type == pygame.KEYDOWN:
			if event.key == pygame.K_UP:
				y_change = -8
			elif event.key == pygame.K_DOWN:
				y_change = 8
			elif event.key == pygame.K_SPACE:
				x_travel = x_shot
				shotsX.append(x_shot)
				shotsY.append(y_shot)
				#unShot = pygame.mixer.Sound('shot.mp3')
				#gunShot.play()

		if event.type == pygame.KEYUP:
			if event.key == pygame.K_UP or event.key == pygame.K_DOWN:
			 	y_change=0

	y = y + y_change
	y_shot = y + 58

	gameDisplay.fill((255,0,0))
	gameDisplay.blit(contraImg,(x,y))
	pygame.draw.circle(gameDisplay,(0,0,255),(x_shot,y_shot),10)
	
	for i in range(len(shotsY)):	
		pygame.draw.circle(gameDisplay,(0,0,255),(shotsX[i],shotsY[i]),10)
		shotsX[i] = shotsX[i] + 8

	if loop % (180/difficulty) == 0:
		enemyY.append(random.randint(0,display_height - enemy_height))
		enemyX.append(display_width)

	for i in range(len(enemyY)):
		gameDisplay.blit(enemyImg,(enemyX[i],enemyY[i]))
		enemyX[i] = enemyX[i] - difficulty * 1.5
		if enemyX[i]<=0 and enemyX[i]>-1.5:
			score = score - 5
		if enemyY[i]>y - enemy_height and enemyY[i]<y+contra_height and enemyX[i]<x + contra_width + 5 and enemyX[i]>x + contra_width - 5:
			health = health - 20
			score = score - 5
			enemyX[i] = -100
		print(score)
		print(health)

		if len(enemyY)>750:
			iter1 = 750
		if len(shotsY)>110:
			iter2 = 110

	for i in range(iter1,len(enemyY)):
		for j in range(iter2,len(shotsX)):
			if shotsY[j]>enemyY[i] and shotsY[j]<enemyY[i] + enemy_height and shotsX[j]<enemyX[i] + 12 and shotsX[j]>enemyX[i] - 4:
				shotsX[j] = display_width + 30
				enemyX[i] = -100
				score = score + 10
				print(score)
				#shotCollide.play()

	pygame.display.update()
	clock.tick(60)

	if score >= 80:
		difficulty = 3
	if score >= 200:
		difficulty = 5
	if score >= 350:
		difficulty = 7
	if score >= 450:
		difficulty = 9

	if health == 0:
		gameExit = True

	text = font.render('SCORE : '+str(score)+'\n', True, (255,255,255))
	gameDisplay.blit(text,(400,400))

	loop = loop + 1

pygame.quit()
quit()





# Instantiate memory
memory = Memory(max_size = memory_size)
for i in range(pretrain_length):
    # If it's the first step
    if i == 0:
        state = Reset()
        
        state, stacked_frames = stack_frames(stacked_frames, state, True)
        
    # Get the next_state, the rewards, done by taking a random action
    choice = random.randint(1,len(possible_actions))-1
    action = possible_actions[choice]
    next_state, reward, done = Step(action)
    
    #env.render()
    
    # Stack the frames
    next_state, stacked_frames = stack_frames(stacked_frames, next_state, False)
    
    
    # If the episode is finished (we're dead 3x)
    if done and deathcount == 3:
        # We finished the episode
        next_state = np.zeros(state.shape)
        #deathcount = 0
        # Add experience to memory
        memory.add((state, action, reward, next_state, done))
        
        # Start a new episode
        state = Reset()
        
        # Stack the frames
        state, stacked_frames = stack_frames(stacked_frames, state, True)
        
    else:
        # Add experience to memory
        memory.add((state, action, reward, next_state, done))
        
        # Our new state is now the next_state
        state = next_state


# In[ ]:





# In[19]:




# Setup TensorBoard Writer
writer = tf.summary.FileWriter("~/tensorboard/1")

## Losses
tf.summary.scalar("Loss", DQNetwork.loss)

write_op = tf.summary.merge_all()


# In[ ]:


# Saver will help us to save our model
saver = tf.train.Saver()

if training == True:
    with tf.Session() as sess:
        # Initialize the variables
        sess.run(tf.global_variables_initializer())

        # Initialize the decay rate (that will use to reduce epsilon) 
        decay_step = 0
        
        rewards_list = []
        for episode in range(total_episodes):
            # Set step to 0
            step = 0
            
            # Initialize the rewards of the episode
            episode_rewards = []
            
            # Make a new episode and observe the first state
            state = Reset()
            
            # Remember that stack frame function also call our preprocess function.
            state, stacked_frames = stack_frames(stacked_frames, state, True)
            
            while step < max_steps:
                step += 1
                
                #Increase decay_step
                decay_step +=1
                
                # Predict the action to take and take it
                action, explore_probability = predict_action(explore_start, explore_stop, decay_rate, decay_step, state, possible_actions)
                
                #Perform the action and get the next_state, reward, and done information
                next_state, reward, done = Step(action)
                
              #  if episode_render:
              #      env.render()
                
                # Add the reward to total reward
                episode_rewards.append(reward)
                
                # If the game is finished
                if done and deathcount == 3:
                    deathcount = 0
                    # The episode ends so no next state
                    next_state = np.zeros((110,84), dtype=np.int)
                    
                    next_state, stacked_frames = stack_frames(stacked_frames, next_state, False)

                    # Set step = max_steps to end the episode
                    step = max_steps

                    # Get the total reward of the episode
                    total_reward = np.sum(episode_rewards)

                    print('Episode: {}'.format(episode),
                                  'Total reward: {}'.format(total_reward),
                                  'Explore P: {:.4f}'.format(explore_probability),
                                'Training Loss {:.4f}'.format(loss))

                    rewards_list.append((episode, total_reward))

                    # Store transition <st,at,rt+1,st+1> in memory D
                    memory.add((state, action, reward, next_state, done))

                else:
                    # Stack the frame of the next_state
                    next_state, stacked_frames = stack_frames(stacked_frames, next_state, False)
                
                    # Add experience to memory
                    memory.add((state, action, reward, next_state, done))

                    # st+1 is now our current state
                    state = next_state
                    

                ### LEARNING PART            
                # Obtain random mini-batch from memory
                batch = memory.sample(batch_size)
                states_mb = np.array([each[0] for each in batch], ndmin=3)
                actions_mb = np.array([each[1] for each in batch])
                rewards_mb = np.array([each[2] for each in batch]) 
                next_states_mb = np.array([each[3] for each in batch], ndmin=3)
                dones_mb = np.array([each[4] for each in batch])

                target_Qs_batch = []

                # Get Q values for next_state 
                Qs_next_state = sess.run(DQNetwork.output, feed_dict = {DQNetwork.inputs_: next_states_mb})
                
                # Set Q_target = r if the episode ends at s+1, otherwise set Q_target = r + gamma*maxQ(s', a')
                for i in range(0, len(batch)):
                    terminal = dones_mb[i]

                    # If we are in a terminal state, only equals reward
                    if terminal:
                        target_Qs_batch.append(rewards_mb[i])
                        
                    else:
                        target = rewards_mb[i] + gamma * np.max(Qs_next_state[i])
                        target_Qs_batch.append(target)
                        

                targets_mb = np.array([each for each in target_Qs_batch])

                loss, _ = sess.run([DQNetwork.loss, DQNetwork.optimizer],
                                        feed_dict={DQNetwork.inputs_: states_mb,
                                                   DQNetwork.target_Q: targets_mb,
                                                   DQNetwork.actions_: actions_mb})

                # Write TF Summaries
        
                summary = sess.run(write_op, feed_dict={DQNetwork.inputs_: states_mb,
                                                       DQNetwork.target_Q: targets_mb,
                                                       DQNetwork.actions_: actions_mb})
                writer.add_summary(summary, episode)
                writer.flush()

            # Save model every 5 episodes
            if episode % 5 == 0:
                save_path = saver.save(sess, "./models/model.ckpt")
                print("Model Saved")


# In[21]:


with tf.Session() as sess:
    total_test_rewards = []
    
    # Load the model
    saver.restore(sess, "./models/model.ckpt")
    
    for episode in range(1):
        total_rewards = 0
        
        state = Reset()
        state, stacked_frames = stack_frames(stacked_frames, state, True)
        
        print("****************************************************")
        print("EPISODE ", episode)
        
        while True:
            # Reshape the state
            state = state.reshape((1, *state_size))
            # Get action from Q-network 
            # Estimate the Qs values state
            Qs = sess.run(DQNetwork.output, feed_dict = {DQNetwork.inputs_: state})
            
            # Take the biggest Q value (= the best action)
            choice = np.argmax(Qs)
            action = possible_actions[choice]
            
            #Perform the action and get the next_state, reward, and done information
            next_state, reward, done = Step(action)
#            env.render()
            
            total_rewards += reward

            if done:
                print ("Score", total_rewards)
                total_test_rewards.append(total_rewards)
                break
                
                
            next_state, stacked_frames = stack_frames(stacked_frames, next_state, False)
			state = next_state