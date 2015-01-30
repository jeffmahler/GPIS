import os
import urllib2
import tarfile


#script to download all the processed meshes in bigbird and then unzip them

pathz="http://rll.berkeley.edu/bigbird/aliases/d1beb9838a/"
mountloc="/mnt/terastation/shape_data/"

def getfiles(obj):
	print 'starting '+obj+' processed'
	if not os.path.exists(mountloc+'BigBIRD/'+obj):
		os.makedirs(mountloc+'BigBIRD/'+obj)
	thirdpath=pathz+"export/"+obj+"/processed.tgz"

	print 'starting '+obj+' processed'
	req=urllib2.Request(thirdpath)
	resp=urllib2.urlopen(req)
	html=resp.read()
	lf=mountloc+'BigBIRD/'+obj+'/processed.tgz'
	lfopen=open(lf,"wb")
	lfopen.write(html)
	lfopen.close()
	toextract=tarfile.open(lf,'r:gz')
	toextract.extractall(mountloc+'BigBIRD/'+obj)
	os.remove(lf)
	os.rename(mountloc+'BigBIRD/'+obj+'/'+obj,mountloc+'BigBIRD/'+obj+'/processed')



getfiles('motts_original_assorted_fruit')
getfiles('nature_valley_crunchy_oats_n_honey')
getfiles('nature_valley_crunchy_variety_pack')
getfiles('nature_valley_soft_baked_oatmeal_squares_peanut_butter')
getfiles('nature_valley_sweet_and_salty_nut_almond')
getfiles('nature_valley_sweet_and_salty_nut_cashew')
getfiles('nature_valley_sweet_and_salty_nut_peanut')
getfiles('nature_valley_sweet_and_salty_nut_roasted_mix_nut')
getfiles('nice_honey_roasted_almonds')
getfiles('nutrigrain_apple_cinnamon')
getfiles('nutrigrain_blueberry')
getfiles('nutrigrain_cherry')
getfiles('nutrigrain_chotolatey_crunch')
getfiles('nutrigrain_fruit_crunch_apple_cobbler')
getfiles('nutrigrain_fruit_crunch_strawberry_parfait')
getfiles('nutrigrain_harvest_blueberry_bliss')
getfiles('nutrigrain_harvest_country_strawberry')
getfiles('nutrigrain_raspberry')
getfiles('nutrigrain_strawberry')
getfiles('nutrigrain_strawberry_greek_yogurt')
getfiles('nutrigrain_toffee_crunch_chocolatey_toffee')
getfiles('palmolive_green')
getfiles('palmolive_orange')
getfiles('paper_cup_holder')
getfiles('paper_plate')
getfiles('pepto_bismol')
getfiles('pop_secret_butter')
getfiles('pop_secret_light_butter')
getfiles('pop_tarts_strawberry')
getfiles('pringles_bbq')
getfiles('progresso_new_england_clam_chowder')
getfiles('quaker_big_chewy_chocolate_chip')
getfiles('quaker_big_chewy_peanut_butter_chocolate_chip')
getfiles('quaker_chewy_chocolate_chip')
getfiles('quaker_chewy_dipps_peanut_butter_chocolate')
getfiles('quaker_chewy_low_fat_chocolate_chunk')
getfiles('quaker_chewy_peanut_butter')
getfiles('quaker_chewy_peanut_butter_chocolate_chip')
getfiles('quaker_chewy_smores')
getfiles('red_bull')
getfiles('red_cup')
getfiles('ritz_crackers')
getfiles('softsoap_clear')
getfiles('softsoap_gold')
getfiles('softsoap_green')
getfiles('softsoap_purple')
getfiles('softsoap_white')
getfiles('south_beach_good_to_go_dark_chocolate')
getfiles('south_beach_good_to_go_peanut_butter')
getfiles('spam')
getfiles('spongebob_squarepants_fruit_snaks')
getfiles('suave_sweet_guava_nectar_body_wash')
getfiles('sunkist_fruit_snacks_mixed_fruit')
getfiles('tapatio_hot_sauce')
getfiles('v8_fusion_peach_mango')
getfiles('v8_fusion_strawberry_banana')
getfiles('vo5_extra_body_volumizing_shampoo')
getfiles('vo5_split_ends_anti_breakage_shampoo')
getfiles('vo5_tea_therapy_healthful_green_tea_smoothing_shampoo')
getfiles('white_rain_sensations_apple_blossom_hydrating_body_wash')
getfiles('white_rain_sensations_ocean_mist_hydrating_body_wash')
getfiles('white_rain_sensations_ocean_mist_hydrating_conditioner')
getfiles('windex')
getfiles('zilla_night_black_heat')

print 'done'