// splits the movie files into seperate timepoints
// saves the timepoints in seperate folder for each position
setBatchMode(true);

// adjust the parameters before the loop and also "czi_name"
folder = "D:\\Kasirer\\experimental_results\\movies\\Utricle\\2021-12-23_E17.5_utricle_atoh_zo\\"; 
tot_pos = 4; // number of positions
pos_final_movie = newArray(12,4,7,12); // an array with the final movies index of each position
tot_movies = 12; // number of movies/files
start_movie = 1;
ts = newArray(8,13,25,28,32,36,30,26,25,38,12); // time points in each movie/file
Ti = 1; // initial t


// make sure that the folders for the positions exist
for (p=1; p<=tot_pos; p++){
	File.makeDirectory(folder + "position" + p); 
}

T = Ti;
for (mi=start_movie; mi<=tot_movies; mi++){ // run over movies
	czi_name = "m"+mi+".czi";
	timepoints = ts[mi-1];
	general_settings = "color_mode=Default rois_import=[ROI manager] specify_range view=Hyperstack stack_order=XYCZT";
	m_positions = 0;
	for (s=1; s<=tot_pos; s++){
		if (mi <= pos_final_movie[s-1]){
			m_positions++;
		}
	}
	for (t=1; t<=timepoints; t++){ // run over timepoints
		s_drift=0;
		for (s=1; s<=tot_pos; s++){ // run over positions
			if (mi > pos_final_movie[s-1]){
				s_drift++;
				continue;
			}
			s_name = "series_"+(s-s_drift)+" ";
			s_str = "_"+(s-s_drift);
			if (m_positions==1){
				s_name = "";
				s_str = "";
			}
			series_settings = s_name+" c_begin"+s_str+"=1 c_end"+s_str+"=2 c_step"+s_str+"=1 z_begin"+s_str+"=1 z_end"+s_str+"=40 z_step"+s_str+"=1 t_begin"+s_str+"="+t+" t_end"+s_str+"="+t+" t_step"+s_str+"=1";
			run("Bio-Formats Importer", "open=["+folder+czi_name+"] "+general_settings+" "+series_settings);
			saveAs("Tiff", folder + "position"+s+"/timepoint"+T+".tif");
			close();
		}
		run("Collect Garbage");
		T++;
	}
}

setBatchMode(false);