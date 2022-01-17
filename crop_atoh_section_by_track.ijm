setBatchMode(true);
// first, track a cell with "point selection" tool and press 'm' to load the measurements window

// update according to the current experiment
timepoints_folder = "D:/Kasirer/experimental_results/movies/Utricle/2021-08-29_p0_utricle_ablation/position2/"; // folder with full timepoints (including z-stack)
savefolder = "D:/Kasirer/experimental_results/movies/Utricle/2021-08-29_p0_utricle_ablation/"; // folder to save in
R = 150; // region radius to crop (in pixels)
switchFZ = 0; // if the time and z dimensions got switched in the results

// create the saving folder if not exist
File.makeDirectory(savefolder); 

getDimensions(width, height, channels, slices, frames);
nR = nResults;
// load selected points to arrays
xs = newArray(nR); 
ys = newArray(nR); 
zs = newArray(nR); 
fs = newArray(nR); 
for (i = 0; i < nR; i++) {
    xs[i] = getResult('X', i);
    ys[i] = getResult('Y', i);
    toUnscaled(xs[i], ys[i]);
    if (switchFZ){
    	zs[i] = getResult("Frame", i);
    	fs[i] = getResult("Slice", i);
    } else {
    	zs[i] = getResult("Slice", i);
    	fs[i] = getResult("Frame", i);	
    }
    //print(xs[i] + " " + ys[i] + " " + zs[i] + " " + fs[i]);
}
ini_frame = fs[0];
fin_frame = fs[nR-1];

// fills in the missing intervals
xpoints = newArray(frames);
ypoints = newArray(frames);
for (i = 0; i < nR-1; i++) {
	xi = xs[i]; xf = xs[i+1]; dx = xf-xi;
	yi = ys[i]; yf = ys[i+1]; dy = yf-yi;
	fi = fs[i]; ff = fs[i+1]; df = ff-fi;
	for (f=fi; f<=ff; f++){
		xpoints[f-1] = round(xi + dx*(f-fi)/df);
		ypoints[f-1] = round(yi + dy*(f-fi)/df);
	}
}
// handles edges
for (f=1; f<=fs[0]; f++){
	xpoints[f-1] = round(xs[0]);
	ypoints[f-1] = round(ys[0]);
}
for (f=fs[nR-1]; f<=frames; f++){
	if (f>=fs[nR-1]){
		xpoints[f-1] = round(xs[nR-1]);
		ypoints[f-1] = round(ys[nR-1]);
	}
}


// open timepoints and crop a section centered at the tracking points
max_slices = 0;
for (f=ini_frame; f<=fin_frame; f++) {
	fname = timepoints_folder + "timepoint" + f + ".tif";
	open(fname);
	rename("tmp1");
	x = xpoints[f-1];
	y = ypoints[f-1];
	if (nSlices > max_slices) {max_slices=nSlices;}
	run("Specify...", "width="+(2*R)+" height="+(2*R)+" x="+(x-R)+" y="+(y-R));
	run("Duplicate...", "duplicate");
	saveAs("Tiff", savefolder + "tp"+ f +".tif");
	rename("tmp2");
	close("tmp2");
	close("tmp1");
	run("Collect Garbage");
}

// combine the timepoints to one movie
newImage("stack_full", "16-bit grayscale-mode", 2*R, 2*R, 2, max_slices, fin_frame-ini_frame+1);
for (f=ini_frame; f<=fin_frame; f++){
	selectWindow("stack_full");
	Stack.setFrame(f-ini_frame+1);
	open(savefolder + "tp"+ f +".tif");
	rename("timepoint");
	getDimensions(www, hhh, channels, nz, fff);
	for (z=1; z<=nz; z++){
		selectImage("timepoint");
		Stack.setSlice(z);
		selectImage("stack_full");
		Stack.setSlice(z);
		for (c=1; c<=channels; c++){
			selectImage("timepoint");
			Stack.setChannel(c);
			run("Select All");
			run("Copy");
			selectImage("stack_full");
			Stack.setChannel(c);
			run("Paste");
		}
	}
	close("timepoint");
}
selectImage("stack_full");
run("Select None");
run("Make Composite");
Stack.setFrame(1);
Stack.setSlice(1);
saveAs("Tiff", savefolder + "full_track1.tif");


setBatchMode(false);