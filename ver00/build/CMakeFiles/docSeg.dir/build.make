# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 2.8

#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:

# Remove some rules from gmake that .SUFFIXES does not remove.
SUFFIXES =

.SUFFIXES: .hpux_make_needs_suffix_list

# Suppress display of executed commands.
$(VERBOSE).SILENT:

# A target that is always out of date.
cmake_force:
.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /usr/bin/cmake

# The command to remove a file.
RM = /usr/bin/cmake -E remove -f

# Escaping for special characters.
EQUALS = =

# The program to use to edit the cache.
CMAKE_EDIT_COMMAND = /usr/bin/ccmake

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /users/jobinkv/2ND_code/wordSegCode/ver00

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /users/jobinkv/2ND_code/wordSegCode/ver00/build

# Include any dependencies generated for this target.
include CMakeFiles/docSeg.dir/depend.make

# Include the progress variables for this target.
include CMakeFiles/docSeg.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/docSeg.dir/flags.make

CMakeFiles/docSeg.dir/main.cpp.o: CMakeFiles/docSeg.dir/flags.make
CMakeFiles/docSeg.dir/main.cpp.o: ../main.cpp
	$(CMAKE_COMMAND) -E cmake_progress_report /users/jobinkv/2ND_code/wordSegCode/ver00/build/CMakeFiles $(CMAKE_PROGRESS_1)
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Building CXX object CMakeFiles/docSeg.dir/main.cpp.o"
	/usr/bin/c++   $(CXX_DEFINES) $(CXX_FLAGS) -o CMakeFiles/docSeg.dir/main.cpp.o -c /users/jobinkv/2ND_code/wordSegCode/ver00/main.cpp

CMakeFiles/docSeg.dir/main.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/docSeg.dir/main.cpp.i"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_FLAGS) -E /users/jobinkv/2ND_code/wordSegCode/ver00/main.cpp > CMakeFiles/docSeg.dir/main.cpp.i

CMakeFiles/docSeg.dir/main.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/docSeg.dir/main.cpp.s"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_FLAGS) -S /users/jobinkv/2ND_code/wordSegCode/ver00/main.cpp -o CMakeFiles/docSeg.dir/main.cpp.s

CMakeFiles/docSeg.dir/main.cpp.o.requires:
.PHONY : CMakeFiles/docSeg.dir/main.cpp.o.requires

CMakeFiles/docSeg.dir/main.cpp.o.provides: CMakeFiles/docSeg.dir/main.cpp.o.requires
	$(MAKE) -f CMakeFiles/docSeg.dir/build.make CMakeFiles/docSeg.dir/main.cpp.o.provides.build
.PHONY : CMakeFiles/docSeg.dir/main.cpp.o.provides

CMakeFiles/docSeg.dir/main.cpp.o.provides.build: CMakeFiles/docSeg.dir/main.cpp.o

CMakeFiles/docSeg.dir/Classifier.cpp.o: CMakeFiles/docSeg.dir/flags.make
CMakeFiles/docSeg.dir/Classifier.cpp.o: ../Classifier.cpp
	$(CMAKE_COMMAND) -E cmake_progress_report /users/jobinkv/2ND_code/wordSegCode/ver00/build/CMakeFiles $(CMAKE_PROGRESS_2)
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Building CXX object CMakeFiles/docSeg.dir/Classifier.cpp.o"
	/usr/bin/c++   $(CXX_DEFINES) $(CXX_FLAGS) -o CMakeFiles/docSeg.dir/Classifier.cpp.o -c /users/jobinkv/2ND_code/wordSegCode/ver00/Classifier.cpp

CMakeFiles/docSeg.dir/Classifier.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/docSeg.dir/Classifier.cpp.i"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_FLAGS) -E /users/jobinkv/2ND_code/wordSegCode/ver00/Classifier.cpp > CMakeFiles/docSeg.dir/Classifier.cpp.i

CMakeFiles/docSeg.dir/Classifier.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/docSeg.dir/Classifier.cpp.s"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_FLAGS) -S /users/jobinkv/2ND_code/wordSegCode/ver00/Classifier.cpp -o CMakeFiles/docSeg.dir/Classifier.cpp.s

CMakeFiles/docSeg.dir/Classifier.cpp.o.requires:
.PHONY : CMakeFiles/docSeg.dir/Classifier.cpp.o.requires

CMakeFiles/docSeg.dir/Classifier.cpp.o.provides: CMakeFiles/docSeg.dir/Classifier.cpp.o.requires
	$(MAKE) -f CMakeFiles/docSeg.dir/build.make CMakeFiles/docSeg.dir/Classifier.cpp.o.provides.build
.PHONY : CMakeFiles/docSeg.dir/Classifier.cpp.o.provides

CMakeFiles/docSeg.dir/Classifier.cpp.o.provides.build: CMakeFiles/docSeg.dir/Classifier.cpp.o

CMakeFiles/docSeg.dir/FeatureComputer.cpp.o: CMakeFiles/docSeg.dir/flags.make
CMakeFiles/docSeg.dir/FeatureComputer.cpp.o: ../FeatureComputer.cpp
	$(CMAKE_COMMAND) -E cmake_progress_report /users/jobinkv/2ND_code/wordSegCode/ver00/build/CMakeFiles $(CMAKE_PROGRESS_3)
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Building CXX object CMakeFiles/docSeg.dir/FeatureComputer.cpp.o"
	/usr/bin/c++   $(CXX_DEFINES) $(CXX_FLAGS) -o CMakeFiles/docSeg.dir/FeatureComputer.cpp.o -c /users/jobinkv/2ND_code/wordSegCode/ver00/FeatureComputer.cpp

CMakeFiles/docSeg.dir/FeatureComputer.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/docSeg.dir/FeatureComputer.cpp.i"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_FLAGS) -E /users/jobinkv/2ND_code/wordSegCode/ver00/FeatureComputer.cpp > CMakeFiles/docSeg.dir/FeatureComputer.cpp.i

CMakeFiles/docSeg.dir/FeatureComputer.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/docSeg.dir/FeatureComputer.cpp.s"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_FLAGS) -S /users/jobinkv/2ND_code/wordSegCode/ver00/FeatureComputer.cpp -o CMakeFiles/docSeg.dir/FeatureComputer.cpp.s

CMakeFiles/docSeg.dir/FeatureComputer.cpp.o.requires:
.PHONY : CMakeFiles/docSeg.dir/FeatureComputer.cpp.o.requires

CMakeFiles/docSeg.dir/FeatureComputer.cpp.o.provides: CMakeFiles/docSeg.dir/FeatureComputer.cpp.o.requires
	$(MAKE) -f CMakeFiles/docSeg.dir/build.make CMakeFiles/docSeg.dir/FeatureComputer.cpp.o.provides.build
.PHONY : CMakeFiles/docSeg.dir/FeatureComputer.cpp.o.provides

CMakeFiles/docSeg.dir/FeatureComputer.cpp.o.provides.build: CMakeFiles/docSeg.dir/FeatureComputer.cpp.o

CMakeFiles/docSeg.dir/HandDetector.cpp.o: CMakeFiles/docSeg.dir/flags.make
CMakeFiles/docSeg.dir/HandDetector.cpp.o: ../HandDetector.cpp
	$(CMAKE_COMMAND) -E cmake_progress_report /users/jobinkv/2ND_code/wordSegCode/ver00/build/CMakeFiles $(CMAKE_PROGRESS_4)
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Building CXX object CMakeFiles/docSeg.dir/HandDetector.cpp.o"
	/usr/bin/c++   $(CXX_DEFINES) $(CXX_FLAGS) -o CMakeFiles/docSeg.dir/HandDetector.cpp.o -c /users/jobinkv/2ND_code/wordSegCode/ver00/HandDetector.cpp

CMakeFiles/docSeg.dir/HandDetector.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/docSeg.dir/HandDetector.cpp.i"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_FLAGS) -E /users/jobinkv/2ND_code/wordSegCode/ver00/HandDetector.cpp > CMakeFiles/docSeg.dir/HandDetector.cpp.i

CMakeFiles/docSeg.dir/HandDetector.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/docSeg.dir/HandDetector.cpp.s"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_FLAGS) -S /users/jobinkv/2ND_code/wordSegCode/ver00/HandDetector.cpp -o CMakeFiles/docSeg.dir/HandDetector.cpp.s

CMakeFiles/docSeg.dir/HandDetector.cpp.o.requires:
.PHONY : CMakeFiles/docSeg.dir/HandDetector.cpp.o.requires

CMakeFiles/docSeg.dir/HandDetector.cpp.o.provides: CMakeFiles/docSeg.dir/HandDetector.cpp.o.requires
	$(MAKE) -f CMakeFiles/docSeg.dir/build.make CMakeFiles/docSeg.dir/HandDetector.cpp.o.provides.build
.PHONY : CMakeFiles/docSeg.dir/HandDetector.cpp.o.provides

CMakeFiles/docSeg.dir/HandDetector.cpp.o.provides.build: CMakeFiles/docSeg.dir/HandDetector.cpp.o

CMakeFiles/docSeg.dir/LcBasic.cpp.o: CMakeFiles/docSeg.dir/flags.make
CMakeFiles/docSeg.dir/LcBasic.cpp.o: ../LcBasic.cpp
	$(CMAKE_COMMAND) -E cmake_progress_report /users/jobinkv/2ND_code/wordSegCode/ver00/build/CMakeFiles $(CMAKE_PROGRESS_5)
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Building CXX object CMakeFiles/docSeg.dir/LcBasic.cpp.o"
	/usr/bin/c++   $(CXX_DEFINES) $(CXX_FLAGS) -o CMakeFiles/docSeg.dir/LcBasic.cpp.o -c /users/jobinkv/2ND_code/wordSegCode/ver00/LcBasic.cpp

CMakeFiles/docSeg.dir/LcBasic.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/docSeg.dir/LcBasic.cpp.i"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_FLAGS) -E /users/jobinkv/2ND_code/wordSegCode/ver00/LcBasic.cpp > CMakeFiles/docSeg.dir/LcBasic.cpp.i

CMakeFiles/docSeg.dir/LcBasic.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/docSeg.dir/LcBasic.cpp.s"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_FLAGS) -S /users/jobinkv/2ND_code/wordSegCode/ver00/LcBasic.cpp -o CMakeFiles/docSeg.dir/LcBasic.cpp.s

CMakeFiles/docSeg.dir/LcBasic.cpp.o.requires:
.PHONY : CMakeFiles/docSeg.dir/LcBasic.cpp.o.requires

CMakeFiles/docSeg.dir/LcBasic.cpp.o.provides: CMakeFiles/docSeg.dir/LcBasic.cpp.o.requires
	$(MAKE) -f CMakeFiles/docSeg.dir/build.make CMakeFiles/docSeg.dir/LcBasic.cpp.o.provides.build
.PHONY : CMakeFiles/docSeg.dir/LcBasic.cpp.o.provides

CMakeFiles/docSeg.dir/LcBasic.cpp.o.provides.build: CMakeFiles/docSeg.dir/LcBasic.cpp.o

CMakeFiles/docSeg.dir/GCoptimization.cpp.o: CMakeFiles/docSeg.dir/flags.make
CMakeFiles/docSeg.dir/GCoptimization.cpp.o: ../GCoptimization.cpp
	$(CMAKE_COMMAND) -E cmake_progress_report /users/jobinkv/2ND_code/wordSegCode/ver00/build/CMakeFiles $(CMAKE_PROGRESS_6)
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Building CXX object CMakeFiles/docSeg.dir/GCoptimization.cpp.o"
	/usr/bin/c++   $(CXX_DEFINES) $(CXX_FLAGS) -o CMakeFiles/docSeg.dir/GCoptimization.cpp.o -c /users/jobinkv/2ND_code/wordSegCode/ver00/GCoptimization.cpp

CMakeFiles/docSeg.dir/GCoptimization.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/docSeg.dir/GCoptimization.cpp.i"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_FLAGS) -E /users/jobinkv/2ND_code/wordSegCode/ver00/GCoptimization.cpp > CMakeFiles/docSeg.dir/GCoptimization.cpp.i

CMakeFiles/docSeg.dir/GCoptimization.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/docSeg.dir/GCoptimization.cpp.s"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_FLAGS) -S /users/jobinkv/2ND_code/wordSegCode/ver00/GCoptimization.cpp -o CMakeFiles/docSeg.dir/GCoptimization.cpp.s

CMakeFiles/docSeg.dir/GCoptimization.cpp.o.requires:
.PHONY : CMakeFiles/docSeg.dir/GCoptimization.cpp.o.requires

CMakeFiles/docSeg.dir/GCoptimization.cpp.o.provides: CMakeFiles/docSeg.dir/GCoptimization.cpp.o.requires
	$(MAKE) -f CMakeFiles/docSeg.dir/build.make CMakeFiles/docSeg.dir/GCoptimization.cpp.o.provides.build
.PHONY : CMakeFiles/docSeg.dir/GCoptimization.cpp.o.provides

CMakeFiles/docSeg.dir/GCoptimization.cpp.o.provides.build: CMakeFiles/docSeg.dir/GCoptimization.cpp.o

CMakeFiles/docSeg.dir/LinkedBlockList.cpp.o: CMakeFiles/docSeg.dir/flags.make
CMakeFiles/docSeg.dir/LinkedBlockList.cpp.o: ../LinkedBlockList.cpp
	$(CMAKE_COMMAND) -E cmake_progress_report /users/jobinkv/2ND_code/wordSegCode/ver00/build/CMakeFiles $(CMAKE_PROGRESS_7)
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Building CXX object CMakeFiles/docSeg.dir/LinkedBlockList.cpp.o"
	/usr/bin/c++   $(CXX_DEFINES) $(CXX_FLAGS) -o CMakeFiles/docSeg.dir/LinkedBlockList.cpp.o -c /users/jobinkv/2ND_code/wordSegCode/ver00/LinkedBlockList.cpp

CMakeFiles/docSeg.dir/LinkedBlockList.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/docSeg.dir/LinkedBlockList.cpp.i"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_FLAGS) -E /users/jobinkv/2ND_code/wordSegCode/ver00/LinkedBlockList.cpp > CMakeFiles/docSeg.dir/LinkedBlockList.cpp.i

CMakeFiles/docSeg.dir/LinkedBlockList.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/docSeg.dir/LinkedBlockList.cpp.s"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_FLAGS) -S /users/jobinkv/2ND_code/wordSegCode/ver00/LinkedBlockList.cpp -o CMakeFiles/docSeg.dir/LinkedBlockList.cpp.s

CMakeFiles/docSeg.dir/LinkedBlockList.cpp.o.requires:
.PHONY : CMakeFiles/docSeg.dir/LinkedBlockList.cpp.o.requires

CMakeFiles/docSeg.dir/LinkedBlockList.cpp.o.provides: CMakeFiles/docSeg.dir/LinkedBlockList.cpp.o.requires
	$(MAKE) -f CMakeFiles/docSeg.dir/build.make CMakeFiles/docSeg.dir/LinkedBlockList.cpp.o.provides.build
.PHONY : CMakeFiles/docSeg.dir/LinkedBlockList.cpp.o.provides

CMakeFiles/docSeg.dir/LinkedBlockList.cpp.o.provides.build: CMakeFiles/docSeg.dir/LinkedBlockList.cpp.o

CMakeFiles/docSeg.dir/rectprior.cpp.o: CMakeFiles/docSeg.dir/flags.make
CMakeFiles/docSeg.dir/rectprior.cpp.o: ../rectprior.cpp
	$(CMAKE_COMMAND) -E cmake_progress_report /users/jobinkv/2ND_code/wordSegCode/ver00/build/CMakeFiles $(CMAKE_PROGRESS_8)
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Building CXX object CMakeFiles/docSeg.dir/rectprior.cpp.o"
	/usr/bin/c++   $(CXX_DEFINES) $(CXX_FLAGS) -o CMakeFiles/docSeg.dir/rectprior.cpp.o -c /users/jobinkv/2ND_code/wordSegCode/ver00/rectprior.cpp

CMakeFiles/docSeg.dir/rectprior.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/docSeg.dir/rectprior.cpp.i"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_FLAGS) -E /users/jobinkv/2ND_code/wordSegCode/ver00/rectprior.cpp > CMakeFiles/docSeg.dir/rectprior.cpp.i

CMakeFiles/docSeg.dir/rectprior.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/docSeg.dir/rectprior.cpp.s"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_FLAGS) -S /users/jobinkv/2ND_code/wordSegCode/ver00/rectprior.cpp -o CMakeFiles/docSeg.dir/rectprior.cpp.s

CMakeFiles/docSeg.dir/rectprior.cpp.o.requires:
.PHONY : CMakeFiles/docSeg.dir/rectprior.cpp.o.requires

CMakeFiles/docSeg.dir/rectprior.cpp.o.provides: CMakeFiles/docSeg.dir/rectprior.cpp.o.requires
	$(MAKE) -f CMakeFiles/docSeg.dir/build.make CMakeFiles/docSeg.dir/rectprior.cpp.o.provides.build
.PHONY : CMakeFiles/docSeg.dir/rectprior.cpp.o.provides

CMakeFiles/docSeg.dir/rectprior.cpp.o.provides.build: CMakeFiles/docSeg.dir/rectprior.cpp.o

# Object files for target docSeg
docSeg_OBJECTS = \
"CMakeFiles/docSeg.dir/main.cpp.o" \
"CMakeFiles/docSeg.dir/Classifier.cpp.o" \
"CMakeFiles/docSeg.dir/FeatureComputer.cpp.o" \
"CMakeFiles/docSeg.dir/HandDetector.cpp.o" \
"CMakeFiles/docSeg.dir/LcBasic.cpp.o" \
"CMakeFiles/docSeg.dir/GCoptimization.cpp.o" \
"CMakeFiles/docSeg.dir/LinkedBlockList.cpp.o" \
"CMakeFiles/docSeg.dir/rectprior.cpp.o"

# External object files for target docSeg
docSeg_EXTERNAL_OBJECTS =

docSeg: CMakeFiles/docSeg.dir/main.cpp.o
docSeg: CMakeFiles/docSeg.dir/Classifier.cpp.o
docSeg: CMakeFiles/docSeg.dir/FeatureComputer.cpp.o
docSeg: CMakeFiles/docSeg.dir/HandDetector.cpp.o
docSeg: CMakeFiles/docSeg.dir/LcBasic.cpp.o
docSeg: CMakeFiles/docSeg.dir/GCoptimization.cpp.o
docSeg: CMakeFiles/docSeg.dir/LinkedBlockList.cpp.o
docSeg: CMakeFiles/docSeg.dir/rectprior.cpp.o
docSeg: CMakeFiles/docSeg.dir/build.make
docSeg: /usr/lib/x86_64-linux-gnu/libopencv_videostab.so.2.4.8
docSeg: /usr/lib/x86_64-linux-gnu/libopencv_video.so.2.4.8
docSeg: /usr/lib/x86_64-linux-gnu/libopencv_ts.so.2.4.8
docSeg: /usr/lib/x86_64-linux-gnu/libopencv_superres.so.2.4.8
docSeg: /usr/lib/x86_64-linux-gnu/libopencv_stitching.so.2.4.8
docSeg: /usr/lib/x86_64-linux-gnu/libopencv_photo.so.2.4.8
docSeg: /usr/lib/x86_64-linux-gnu/libopencv_ocl.so.2.4.8
docSeg: /usr/lib/x86_64-linux-gnu/libopencv_objdetect.so.2.4.8
docSeg: /usr/lib/x86_64-linux-gnu/libopencv_nonfree.so.2.4.8
docSeg: /usr/lib/x86_64-linux-gnu/libopencv_ml.so.2.4.8
docSeg: /usr/lib/x86_64-linux-gnu/libopencv_legacy.so.2.4.8
docSeg: /usr/lib/x86_64-linux-gnu/libopencv_imgproc.so.2.4.8
docSeg: /usr/lib/x86_64-linux-gnu/libopencv_highgui.so.2.4.8
docSeg: /usr/lib/x86_64-linux-gnu/libopencv_gpu.so.2.4.8
docSeg: /usr/lib/x86_64-linux-gnu/libopencv_flann.so.2.4.8
docSeg: /usr/lib/x86_64-linux-gnu/libopencv_features2d.so.2.4.8
docSeg: /usr/lib/x86_64-linux-gnu/libopencv_core.so.2.4.8
docSeg: /usr/lib/x86_64-linux-gnu/libopencv_contrib.so.2.4.8
docSeg: /usr/lib/x86_64-linux-gnu/libopencv_calib3d.so.2.4.8
docSeg: /usr/lib/x86_64-linux-gnu/libopencv_nonfree.so.2.4.8
docSeg: /usr/lib/x86_64-linux-gnu/libopencv_ocl.so.2.4.8
docSeg: /usr/lib/x86_64-linux-gnu/libopencv_gpu.so.2.4.8
docSeg: /usr/lib/x86_64-linux-gnu/libopencv_photo.so.2.4.8
docSeg: /usr/lib/x86_64-linux-gnu/libopencv_objdetect.so.2.4.8
docSeg: /usr/lib/x86_64-linux-gnu/libopencv_legacy.so.2.4.8
docSeg: /usr/lib/x86_64-linux-gnu/libopencv_video.so.2.4.8
docSeg: /usr/lib/x86_64-linux-gnu/libopencv_ml.so.2.4.8
docSeg: /usr/lib/x86_64-linux-gnu/libopencv_calib3d.so.2.4.8
docSeg: /usr/lib/x86_64-linux-gnu/libopencv_features2d.so.2.4.8
docSeg: /usr/lib/x86_64-linux-gnu/libopencv_highgui.so.2.4.8
docSeg: /usr/lib/x86_64-linux-gnu/libopencv_imgproc.so.2.4.8
docSeg: /usr/lib/x86_64-linux-gnu/libopencv_flann.so.2.4.8
docSeg: /usr/lib/x86_64-linux-gnu/libopencv_core.so.2.4.8
docSeg: CMakeFiles/docSeg.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --red --bold "Linking CXX executable docSeg"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/docSeg.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/docSeg.dir/build: docSeg
.PHONY : CMakeFiles/docSeg.dir/build

CMakeFiles/docSeg.dir/requires: CMakeFiles/docSeg.dir/main.cpp.o.requires
CMakeFiles/docSeg.dir/requires: CMakeFiles/docSeg.dir/Classifier.cpp.o.requires
CMakeFiles/docSeg.dir/requires: CMakeFiles/docSeg.dir/FeatureComputer.cpp.o.requires
CMakeFiles/docSeg.dir/requires: CMakeFiles/docSeg.dir/HandDetector.cpp.o.requires
CMakeFiles/docSeg.dir/requires: CMakeFiles/docSeg.dir/LcBasic.cpp.o.requires
CMakeFiles/docSeg.dir/requires: CMakeFiles/docSeg.dir/GCoptimization.cpp.o.requires
CMakeFiles/docSeg.dir/requires: CMakeFiles/docSeg.dir/LinkedBlockList.cpp.o.requires
CMakeFiles/docSeg.dir/requires: CMakeFiles/docSeg.dir/rectprior.cpp.o.requires
.PHONY : CMakeFiles/docSeg.dir/requires

CMakeFiles/docSeg.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/docSeg.dir/cmake_clean.cmake
.PHONY : CMakeFiles/docSeg.dir/clean

CMakeFiles/docSeg.dir/depend:
	cd /users/jobinkv/2ND_code/wordSegCode/ver00/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /users/jobinkv/2ND_code/wordSegCode/ver00 /users/jobinkv/2ND_code/wordSegCode/ver00 /users/jobinkv/2ND_code/wordSegCode/ver00/build /users/jobinkv/2ND_code/wordSegCode/ver00/build /users/jobinkv/2ND_code/wordSegCode/ver00/build/CMakeFiles/docSeg.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/docSeg.dir/depend

