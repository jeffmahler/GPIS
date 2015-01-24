import os
import urllib2
import tarfile

#script for downloading everything from the YCB database

pathz="http://rll.berkeley.edu/ycb/export/"
mountloc="/mnt/terastation/shape_data/"

def getfiles(obj):
	print 'starting '+obj
	#if not os.path.exists(mountloc+'YCB/'+obj):
	#	os.makedirs(mountloc+'YCB/'+obj)
	thirdpath=pathz+obj+".tgz"

	req=urllib2.Request(thirdpath)
	resp=urllib2.urlopen(req)
	html=resp.read()
	lf=mountloc+'YCB/zipped.tgz'
	lfopen=open(lf,"wb")
	lfopen.write(html)
	lfopen.close()
	print 'unzipping '+obj
	toextract=tarfile.open(lf,'r:gz')
	toextract.extractall(mountloc+'YCB')
	os.remove(lf)
	



getfiles('clorox_disinfecting_wipes_35')
getfiles('comet_lemon_fresh_bleach')
getfiles('dark_red_foam_block_with_three_holes')
getfiles('domino_sugar_1lb')
getfiles('expo_black_dry_erase_marker')
getfiles('expo_black_dry_erase_marker_fine')
getfiles('extra_small_black_spring_clamp')
getfiles('frenchs_classic_yellow_mustard_14oz')
getfiles('jell-o_chocolate_flavor_pudding')
getfiles('jell-o_strawberry_gelatin_dessert')
getfiles('large_black_spring_clamp')
getfiles('learning_resources_one-inch_color_cubes_box')
getfiles('master_chef_ground_coffee_297g')
getfiles('medium_black_spring_clamp')
getfiles('melissa_doug_farm_fresh_fruit_apple')
getfiles('melissa_doug_farm_fresh_fruit_banana')
getfiles('melissa_doug_farm_fresh_fruit_lemon')
getfiles('melissa_doug_farm_fresh_fruit_orange')
getfiles('melissa_doug_farm_fresh_fruit_peach')
getfiles('melissa_doug_farm_fresh_fruit_pear')
getfiles('melissa_doug_farm_fresh_fruit_plum')
getfiles('melissa_doug_farm_fresh_fruit_strawberry')
getfiles('melissa_doug_play-time_produce_farm_fresh_fruit_unopened_box')
getfiles('morton_pepper_shaker')
getfiles('morton_salt_shaker')
getfiles('moutain_security_steel_shackle')
getfiles('moutain_security_steel_shackle_key')
getfiles('orange_wood_block_1inx1in')
getfiles('penn_raquet_ball')
getfiles('plastic_bolt_grey')
getfiles('plastic_nut_grey')
getfiles('plastic_wine_cup')
getfiles('play_go_rainbow_stakin_cups_10_blue')
getfiles('play_go_rainbow_stakin_cups_1_yellow')
getfiles('play_go_rainbow_stakin_cups_2_orange')
getfiles('play_go_rainbow_stakin_cups_3_red')
getfiles('play_go_rainbow_stakin_cups_5_green')
getfiles('play_go_rainbow_stakin_cups_6_purple')
getfiles('play_go_rainbow_stakin_cups_7_yellow')
getfiles('play_go_rainbow_stakin_cups_8_orange')
getfiles('play_go_rainbow_stakin_cups_9_red')
getfiles('play_go_rainbow_stakin_cups_blue_4')
getfiles('play_go_rainbow_stakin_cups_box')
getfiles('pringles_original')
getfiles('purple_wood_block_1inx1in')
getfiles('red_metal_bowl_white_speckles')
getfiles('red_metal_cup_white_speckles')
getfiles('red_metal_plate_white_speckles')
getfiles('red_wood_block_1inx1in')
getfiles('rubbermaid_ice_guard_pitcher_blue')
getfiles('sharpie_marker')
getfiles('small_black_spring_clamp')
getfiles('soft_scrub_2lb_4oz')
getfiles('spam_12oz')
getfiles('sponge_with_textured_cover')
getfiles('stainless_steel_fork_red_handle')
getfiles('stainless_steel_knife_red_handle')
getfiles('stainless_steel_spatula')
getfiles('stainless_steel_spoon_red_handle')
getfiles('stanley_13oz_hammer')
getfiles('stanley_flathead_screwdriver')
getfiles('stanley_philips_screwdriver')
getfiles('starkist_chunk_light_tuna')
getfiles('sterilite_bin_12qt_bottom')
getfiles('sterilite_bin_12qt_cap')
getfiles('thick_wood_block_6in')
getfiles('wearever_cooking_pan_with_lid')
getfiles('wescott_orange_grey_scissors')
getfiles('white_rope')
getfiles('wilson_100_tennis_ball')
getfiles('wilson_golf_ball')
getfiles('windex')
getfiles('yellow_plastic_chain')
getfiles('yellow_wood_block_1inx1in')



print 'done'