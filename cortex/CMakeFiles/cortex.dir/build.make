# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.3

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:


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
CMAKE_COMMAND = /usr/local/Cellar/cmake/3.3.2/bin/cmake

# The command to remove a file.
RM = /usr/local/Cellar/cmake/3.3.2/bin/cmake -E remove -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /Users/guillaume/Dropbox/ICL-2014/Code/C-Code/cortex/cortex

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /Users/guillaume/Dropbox/ICL-2014/Code/C-Code/cortex/cortex

# Include any dependencies generated for this target.
include CMakeFiles/cortex.dir/depend.make

# Include the progress variables for this target.
include CMakeFiles/cortex.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/cortex.dir/flags.make

CMakeFiles/cortex.dir/main.cpp.o: CMakeFiles/cortex.dir/flags.make
CMakeFiles/cortex.dir/main.cpp.o: main.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/Users/guillaume/Dropbox/ICL-2014/Code/C-Code/cortex/cortex/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object CMakeFiles/cortex.dir/main.cpp.o"
	/Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/bin/c++   $(CXX_DEFINES) $(CXX_FLAGS) -o CMakeFiles/cortex.dir/main.cpp.o -c /Users/guillaume/Dropbox/ICL-2014/Code/C-Code/cortex/cortex/main.cpp

CMakeFiles/cortex.dir/main.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/cortex.dir/main.cpp.i"
	/Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/bin/c++  $(CXX_DEFINES) $(CXX_FLAGS) -E /Users/guillaume/Dropbox/ICL-2014/Code/C-Code/cortex/cortex/main.cpp > CMakeFiles/cortex.dir/main.cpp.i

CMakeFiles/cortex.dir/main.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/cortex.dir/main.cpp.s"
	/Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/bin/c++  $(CXX_DEFINES) $(CXX_FLAGS) -S /Users/guillaume/Dropbox/ICL-2014/Code/C-Code/cortex/cortex/main.cpp -o CMakeFiles/cortex.dir/main.cpp.s

CMakeFiles/cortex.dir/main.cpp.o.requires:

.PHONY : CMakeFiles/cortex.dir/main.cpp.o.requires

CMakeFiles/cortex.dir/main.cpp.o.provides: CMakeFiles/cortex.dir/main.cpp.o.requires
	$(MAKE) -f CMakeFiles/cortex.dir/build.make CMakeFiles/cortex.dir/main.cpp.o.provides.build
.PHONY : CMakeFiles/cortex.dir/main.cpp.o.provides

CMakeFiles/cortex.dir/main.cpp.o.provides.build: CMakeFiles/cortex.dir/main.cpp.o


CMakeFiles/cortex.dir/utils.cpp.o: CMakeFiles/cortex.dir/flags.make
CMakeFiles/cortex.dir/utils.cpp.o: utils.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/Users/guillaume/Dropbox/ICL-2014/Code/C-Code/cortex/cortex/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Building CXX object CMakeFiles/cortex.dir/utils.cpp.o"
	/Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/bin/c++   $(CXX_DEFINES) $(CXX_FLAGS) -o CMakeFiles/cortex.dir/utils.cpp.o -c /Users/guillaume/Dropbox/ICL-2014/Code/C-Code/cortex/cortex/utils.cpp

CMakeFiles/cortex.dir/utils.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/cortex.dir/utils.cpp.i"
	/Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/bin/c++  $(CXX_DEFINES) $(CXX_FLAGS) -E /Users/guillaume/Dropbox/ICL-2014/Code/C-Code/cortex/cortex/utils.cpp > CMakeFiles/cortex.dir/utils.cpp.i

CMakeFiles/cortex.dir/utils.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/cortex.dir/utils.cpp.s"
	/Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/bin/c++  $(CXX_DEFINES) $(CXX_FLAGS) -S /Users/guillaume/Dropbox/ICL-2014/Code/C-Code/cortex/cortex/utils.cpp -o CMakeFiles/cortex.dir/utils.cpp.s

CMakeFiles/cortex.dir/utils.cpp.o.requires:

.PHONY : CMakeFiles/cortex.dir/utils.cpp.o.requires

CMakeFiles/cortex.dir/utils.cpp.o.provides: CMakeFiles/cortex.dir/utils.cpp.o.requires
	$(MAKE) -f CMakeFiles/cortex.dir/build.make CMakeFiles/cortex.dir/utils.cpp.o.provides.build
.PHONY : CMakeFiles/cortex.dir/utils.cpp.o.provides

CMakeFiles/cortex.dir/utils.cpp.o.provides.build: CMakeFiles/cortex.dir/utils.cpp.o


# Object files for target cortex
cortex_OBJECTS = \
"CMakeFiles/cortex.dir/main.cpp.o" \
"CMakeFiles/cortex.dir/utils.cpp.o"

# External object files for target cortex
cortex_EXTERNAL_OBJECTS =

cortex: CMakeFiles/cortex.dir/main.cpp.o
cortex: CMakeFiles/cortex.dir/utils.cpp.o
cortex: CMakeFiles/cortex.dir/build.make
cortex: CMakeFiles/cortex.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/Users/guillaume/Dropbox/ICL-2014/Code/C-Code/cortex/cortex/CMakeFiles --progress-num=$(CMAKE_PROGRESS_3) "Linking CXX executable cortex"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/cortex.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/cortex.dir/build: cortex

.PHONY : CMakeFiles/cortex.dir/build

CMakeFiles/cortex.dir/requires: CMakeFiles/cortex.dir/main.cpp.o.requires
CMakeFiles/cortex.dir/requires: CMakeFiles/cortex.dir/utils.cpp.o.requires

.PHONY : CMakeFiles/cortex.dir/requires

CMakeFiles/cortex.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/cortex.dir/cmake_clean.cmake
.PHONY : CMakeFiles/cortex.dir/clean

CMakeFiles/cortex.dir/depend:
	cd /Users/guillaume/Dropbox/ICL-2014/Code/C-Code/cortex/cortex && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /Users/guillaume/Dropbox/ICL-2014/Code/C-Code/cortex/cortex /Users/guillaume/Dropbox/ICL-2014/Code/C-Code/cortex/cortex /Users/guillaume/Dropbox/ICL-2014/Code/C-Code/cortex/cortex /Users/guillaume/Dropbox/ICL-2014/Code/C-Code/cortex/cortex /Users/guillaume/Dropbox/ICL-2014/Code/C-Code/cortex/cortex/CMakeFiles/cortex.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/cortex.dir/depend

