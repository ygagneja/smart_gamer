import pygame
import random
import numpy as np
import time

model = np.arange(1500000).reshape((500000,3))

learning_rate = 0.8
gamma = 0.99

score = 0
health = 100
loop = 0
difficulty = 1

display_width = 800
display_height = 600
contra_height = 165
contra_width = 108
enemy_height = 80

tempUP = 0
tempDOWN = 0 #to store previous enemy state

gameDisplay = pygame.display.set_mode((display_width,display_height))
pygame.display.set_caption('Contra Revamped')
clock = pygame.time.Clock()

contraImg = pygame.image.load('contra.png')
enemyImg = pygame.image.load('enemy.png')

# music = pygame.mixer.music.load('background.mp3')
# pygame.mixer.music.play(-1)

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

shoot = False
idx = 0
gameExit = False

QIDic = {}

Q = np.random.rand(500000,3)



def new_state_after_action(s,act):
	if act == 2:
		if s[0] + 58 + 8 > display_height:
			return [display_height - 58,s[1],s[2]]
		return [s[0]+8,s[1],s[2]]
	elif act == 1:
		if s[0] + 58 - 8 < 0:
			return [-58,s[1],s[2]]
		return [s[0]-8,s[1],s[2]]
	else:
		return [s[0],s[1],s[2]]

def state_to_number(s):
	e = s[1]
	c = s[0]
	n = s[1] * s[0]

	if n in QIDic:
		return QIDic[n]
	else:
		if len(QIDic):
			maximum = max(QIDic,key = QIDic.get)
			QIDic[n] = QIDic[maximum] + 1
		else:
			QIDic[n] = 1

	return QIDic[n]

def get_best_action(s,y_shot):
	if y_shot + 20 > display_height:
		return 1
	if y_shot - 20 < 0:
		return 2 
	return np.argmax(Q[state_to_number(s),:])

def get_reward(s):
	if s[0] + 58>s[2] and s[0] + 58<s[2] + enemy_height:
		if contra_width + 20 < s[1]:
			reward = 10
		else:
			reward = -15
	elif s[1]<=20:
		reward = int(-10 * ((s[2] - s[0] + 58)/display_height))
	elif s[0] + 58 + 20>display_height or s[0] + 58 - 20:
		reward = -5
	else:
		reward = -3

	return reward





pygame.init()

while not gameExit:

	for event in pygame.event.get():
		if event.type == pygame.QUIT:
			gameExit = True

	if loop % (180/difficulty) == 0:
		enemyY.append(random.randint(0,display_height - enemy_height))
		enemyX.append(display_width)

	#REWARD SYSTEM
	if len(enemyX):
		s = [y,enemyX[0],enemyY[0]]
	else:
		s = [y,display_width,random.randint(0,display_height - enemy_height)]
	
	act = get_best_action(s,y_shot)
	r0 = get_reward(s)
	s1 = new_state_after_action(s,act)
	Q[state_to_number(s),act] += learning_rate * (r0 + np.max(Q[state_to_number(s1), :] * gamma) - Q[state_to_number(s),act])
	#REWARD SYSTEM

	if act == 2 and y_shot + 8<display_height:
		y = y + 8
	elif act == 1 and y_shot - 8>0:
		y = y - 8

	y_shot = y + 58

	gameDisplay.fill((255,0,0))
	gameDisplay.blit(contraImg,(x,y))
	pygame.draw.circle(gameDisplay,(0,0,255),(x_shot,y_shot),10)
	
	for i in range(len(shotsY)):	
		pygame.draw.circle(gameDisplay,(0,0,255),(shotsX[i],shotsY[i]),10)
		shotsX[i] = shotsX[i] + 8

	for i in range(len(enemyY)):
		enemyX[i] = enemyX[i] - difficulty * 1.5
		gameDisplay.blit(enemyImg,(enemyX[i],enemyY[i]))
	
	for i in range(len(enemyY)):
		if y + contra_height>enemyY[i] + enemy_height and enemyY[i]>y and x + contra_width>enemyX[i]:
			health = health - 20
			score = score - 5
			enemyX.pop(i)
			enemyY.pop(i)
			print("Score: "+ str(score))
			print("Health: "+ str(health))
			break

	for i in range(len(enemyY)):
		if enemyX[i]<=0 and enemyX[i]>-1.5:
			score = score - 5
			enemyX.pop(i)
			enemyY.pop(i)
			print("Score: "+ str(score))
			print("Health: "+ str(health))
			break

	for i in range(len(enemyY)):
		for j in range(len(shotsX)):
			if shotsY[j]>enemyY[i] and shotsY[j]<enemyY[i] + enemy_height and shotsX[j]>enemyX[i]:
				enemyX.pop(i)
				enemyY.pop(i)
				shotsX.pop(j)
				shotsY.pop(j)
				score = score + 10
				print("Score: "+ str(score))
				print("Health: "+ str(health))
				break

	for i in range(len(enemyY)):
		if y_shot>enemyY[i] and y_shot<enemyY[i] + enemy_height:
			if contra_width + 10 < enemyX[i]:
				if y_shot>tempUP or y_shot<tempDOWN:
					shoot = True
					idx = i
					shotsX.append(x_shot)
					shotsY.append(y_shot)
					break

	for i in range(len(shotsY)):
		if shotsX[i]>display_width + 20:
			shotsX.pop(i)
			shotsY.pop(i)
			break

	pygame.display.update()
	clock.tick(100000)

	# if score >= 80:
	# 	difficulty = 3
	# if score >= 200:
	# 	difficulty = 5
	# if score >= 350:
	# 	difficulty = 7
	# if score >= 450:
	# 	difficulty = 9

	# if health == 0:
	# 	gameExit = True

	loop = loop + 1

	if shoot:
		shoot = False
		tempUP = enemyY[idx] + enemy_height
		tempDOWN = enemyY[idx]

	if loop%250000 == 0 :
		with open('model.txt','w') as outfile:
			for slice_2d in model:
				np.savetxt(outfile,slice_2d)

pygame.quit()
quit()