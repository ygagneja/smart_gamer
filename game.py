import pygame
import random

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

contraImg = pygame.image.load('resources/contra.png')
enemyImg = pygame.image.load('resources/enemy.png')

music = pygame.mixer.music.load('resources/background.mp3')
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
