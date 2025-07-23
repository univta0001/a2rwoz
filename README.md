# a2rwoz (A2R to WOZ Converter)

## What is this?
This is a utility to convert A2R to WOZ. The supported version for A2R is version 1 to version 3. The generated WOZ file is either WOZ 2.0 or 2.1 depending flux tracks are generated or not.

## Usage

- To convert a A2R image to WOZ. 

  a2rwoz input.a2r output.woz

- To convert a A2R image to WOZ with TMAP tracks only

  a2rwoz --woz input.a2r output.woz

- `a2rwoz --help` will display:

      Usage: a2rwoz.exe [OPTIONS] <input.a2r> <output.woz>
      
      Arguments:
        <input.a2r>
        <output.woz>
      
      Options:
            --creator <CREATOR>              Set the creator (max 32 bytes. If less than 32 bytes,
                                             it will be padded with space) [default: A2RWOZ]
            --woz                            Set woz to true to dump tracks in TMAP
            --full-tracks                    Dump all tracks in 0.25 tracks increments
            --duplicate-quarter-tracks       Duplicate to quarter tracks
            --compare-tracks                 Compare tracks
            --flux <FLUX>                    Convert track to flux
            --tmap <TMAP>                    Convert track to tmap
            --bit-timing <BIT_TIMING>        Bit Timing [default: 32]
            --fast-loop                      Only return the first loop on each track
            --enable-fallback                Enable fallback if loop not found
            --tracks <TRACKS>                If loop not found, specified tracks will be added
            --disable-timing                 Disable timing
            --disable-xtiming                Disable xtiming
            --use-fft                        Use fft to find loop if loop not found
            --delete-tracks <DELETE_TRACKS>  Delete specified track onwards
            --show-failed-loop               Show information on loop not found on tracks
            --show-unsolved-tracks           Show tracks that are not solved
            --debug                          Enable debug
        -h, --help                           Print help
        -V, --version                        Print version  

