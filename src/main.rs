use clap::Parser;
use indicatif::{ProgressBar, ProgressStyle};
use rayon::prelude::*;
use rustfft::{FftPlanner, num_complex::Complex};
use std::collections::BTreeSet;
use std::error::Error;
use std::fmt::{Display, Formatter};
use std::io::{IsTerminal, Write};
use std::path::Path;
use std::str::FromStr;
use std::string::FromUtf8Error;
use std::sync::{Arc, Mutex};

type AppError = Box<dyn Error>;

// Constants
const A2R_A2R1_HEADER: u32 = 0x31523241;
const A2R_A2R2_HEADER: u32 = 0x32523241;
const A2R_A2R3_HEADER: u32 = 0x33523241;

const A2R_INFO_CHUNK: u32 = 0x4f464e49;
const A2R_RWCP_CHUNK: u32 = 0x50435752;
const A2R_SLVD_CHUNK: u32 = 0x44564c53;
const A2R_STRM_CHUNK: u32 = 0x4d525453;
const A2R_DATA_CHUNK: u32 = 0x41544144;
const A2R_META_CHUNK: u32 = 0x4154454d;

const WOZ_MAX_TRACKS: usize = 160;
const WOZ_CREATOR_LEN: usize = 32;
const WOZ_BLOCK_SIZE: usize = 512;

const LOOP_POINT_DELTA: u32 = 57000;

/// Maximum allowed mismatch ratio when comparing two tracks
const MAX_MISMATCH_RATIO: f64 = 0.001; // 0.1 %
const ACCURACY_MULTIPLIER: u32 = 10000;
const PERFECT_ACCURACY: u32 = 10000;

const LABEL: &str = "A2RWOZ";

#[derive(Default)]
struct WozTrackEntry {
    loop_found: bool,
    flux_data: Vec<u8>,
    loop_accuracy: u32,
    capture_type: u8,
    original_flux_data: Vec<u8>,
}

fn parse_creator(s: &str) -> Result<String, String> {
    if s.len() > WOZ_CREATOR_LEN {
        Err(format!(
            "Creator cannot exceed 32 characters (got {} characters)",
            s.len()
        ))
    } else {
        Ok(format!("{: <width$}", s, width = 32))
    }
}

#[derive(Parser, Debug, Clone)]
#[command(version, about)]
struct Args {
    /// Set the creator (max 32 bytes. If less than 32 bytes, it will be padded with space)
    #[arg(long, default_value = "A2RWOZ", value_parser = parse_creator)]
    creator: String,

    /// Set woz to true to dump tracks in TMAP
    #[arg(long, default_value_t = false)]
    woz: bool,

    /// Dump all tracks in 0.25 tracks increments
    #[arg(long, default_value_t = false)]
    full_tracks: bool,

    /// Duplicate to quarter tracks
    #[arg(long, default_value_t = false)]
    duplicate_quarter_tracks: bool,

    /// Compare tracks
    #[arg(long, default_value_t = false)]
    compare_tracks: bool,

    /// Convert track to flux
    #[arg(long, value_parser = parse_track_ranges)]
    flux: Option<std::vec::Vec<u8>>,

    /// Convert track to tmap
    #[arg(long, value_parser = parse_track_ranges)]
    tmap: Option<std::vec::Vec<u8>>,

    /// Bit Timing
    #[arg(long, default_value_t = 32)]
    bit_timing: u8,

    /// Only return the first loop on each track
    #[arg(long, default_value_t = false)]
    fast_loop: bool,

    /// Enable fallback if loop not found
    #[arg(long, default_value_t = false)]
    enable_fallback: bool,

    /// If loop not found, specified tracks will be added
    #[arg(long, value_parser = parse_track_ranges)]
    tracks: Option<std::vec::Vec<u8>>,

    /// Disable timing
    #[arg(long, default_value_t = false)]
    disable_timing: bool,

    /// Disable xtiming
    #[arg(long, default_value_t = false)]
    disable_xtiming: bool,

    /// Use fft to find loop if loop not found
    #[arg(long)]
    use_fft: bool,

    /// Delete specified track onwards
    #[arg(long, value_parser = parse_track_ranges)]
    delete_tracks: Option<std::vec::Vec<u8>>,

    /// Show information on loop not found on tracks
    #[arg(long)]
    show_failed_loop: bool,

    /// Show tracks that are not solved
    #[arg(long)]
    show_unsolved_tracks: bool,

    /// Enable KMP algorithm
    #[arg(long)]
    kmp: bool,

    /// Disable parallel
    #[arg(long, default_value_t = false)]
    disable_parallel: bool,

    /// Enable debug
    #[arg(long)]
    debug: bool,

    #[arg(long, hide = true, default_value_t = 125000)]
    resolution: u32,

    #[arg(index = 1, value_name = "input.a2r")]
    input: String,

    #[arg(index = 2, value_name = "output.woz")]
    output: String,
}

#[derive(Default)]
struct Info {
    version: u8,
    creator: String,
    disk_type: u8,
    write_protected: bool,
    synchronized: bool,
    hard_sector_count: u8,
}

impl Display for Info {
    fn fmt(&self, f: &mut Formatter) -> Result<(), std::fmt::Error> {
        let version = match self.version {
            1 => "1 (A2R1)",
            2 => "2 (A2R2)",
            3 => "3 (A2R3)",
            _ => "Unknown",
        };
        let info = format!("{}INFO{}", green_color(false), reset_color(false));
        writeln!(f, "{info}: {:<20}: {version}", "Version")?;

        let disk_type = match self.disk_type {
            1 => "5.25-inch (140K)",
            2 => "3.5-inch",
            _ => "Unknown",
        };
        writeln!(f, "{info}: {:<20}: {disk_type}", "Disk type")?;
        writeln!(
            f,
            "{info}: {:<20}: {}",
            "Write protected", self.write_protected
        )?;
        writeln!(
            f,
            "{info}: {:<20}: {}",
            "Tracks synchronized", self.synchronized
        )?;
        Ok(())
    }
}

#[derive(PartialEq)]
enum Capture {
    Rwcp,
    Slvd,
    Strm,
    Data,
}

macro_rules! a2_debug {
    ($debug:tt) => {
        if $debug {
            eprintln!()
        }
    };
    ($debug:tt,$($arg:tt)*) => {{
        if $debug {
            eprint!("{LABEL} - ");
            eprintln!($($arg)*);
        }
    }};
}

fn green_color(stderr: bool) -> &'static str {
    if is_terminal(stderr) { "\x1b[32m" } else { "" }
}

fn red_color(stderr: bool) -> &'static str {
    if is_terminal(stderr) { "\x1b[31m" } else { "" }
}

fn reset_color(stderr: bool) -> &'static str {
    if is_terminal(stderr) { "\x1b[0m" } else { "" }
}

fn is_terminal(stderr: bool) -> bool {
    if !stderr {
        std::io::stdout().is_terminal()
    } else {
        std::io::stderr().is_terminal()
    }
}

fn read_a2r_u32(dsk: &[u8], offset: usize) -> u32 {
    let mut data = [0u8; 4];
    data.copy_from_slice(&dsk[offset..offset + 4]);
    u32::from_le_bytes(data)
}

fn read_a2r_big_u32(dsk: &[u8], offset: usize) -> u32 {
    let mut data = [0u8; 4];
    data.copy_from_slice(&dsk[offset..offset + 4]);
    u32::from_be_bytes(data)
}

fn write_woz_u32(dsk: &mut Vec<u8>, value: u32) {
    dsk.extend_from_slice(&value.to_le_bytes());
}

fn write_woz_u16(dsk: &mut Vec<u8>, value: u16) {
    dsk.extend_from_slice(&value.to_le_bytes());
}

fn process_info(data: &[u8], offset: usize, info: &mut Info, a2r3: bool) {
    if a2r3 {
        info.version = 3;
    } else {
        info.version = 2;
    }
    info.creator = str::from_utf8(&data[offset + 1..offset + 33])
        .unwrap_or("Unknown")
        .into();
    info.disk_type = data[offset + 33];
    info.write_protected = data[offset + 34] == 1;
    info.synchronized = data[offset + 35] == 1;

    if a2r3 {
        info.hard_sector_count = data[offset + 36];
    }
}

fn get_label(capture: &Capture) -> &'static str {
    match *capture {
        Capture::Strm => "STRM",
        Capture::Data => "DATA",
        Capture::Rwcp => "RWCP",
        Capture::Slvd => "SLVD",
    }
}

fn process_flux_capture<F>(
    data_input: (&[u8], &mut usize, &mut Args),
    capture: &Capture,
    break_condition: u8,
    data_offset: usize,
    read_data_fn: F,
    hard_sector_count: Option<u8>,
) -> Vec<WozTrackEntry>
where
    F: Fn(&[u8], usize, &Capture, Option<u8>) -> (u8, u8, u32, u32, usize),
{
    let (data, offset, args) = data_input;
    let mut woz_track = Vec::new();
    for _ in 0..WOZ_MAX_TRACKS {
        woz_track.push(WozTrackEntry::default());
    }

    let label = get_label(capture);
    let debug = args.debug;

    let estimated_bit_timing = get_speed(data, *offset + data_offset);
    a2_debug!(
        debug,
        "{label}: Estimated bit timing : {}",
        estimated_bit_timing
    );

    if args.bit_timing == 0 {
        args.bit_timing = estimated_bit_timing
    }

    let mut track_entry = Vec::new();
    while *offset < data.len() {
        let current_byte = data[*offset];
        if current_byte == break_condition {
            break;
        }

        let (location, capture_type, length, loop_point, bytes_read) =
            read_data_fn(data, *offset, capture, hard_sector_count);

        *offset += bytes_read;
        track_entry.push((*offset, length, location, capture_type, loop_point));
        *offset += length as usize;
    }

    let bar = create_progress_bar(woz_track.len() as u64, args.debug);
    let mutex_woz_track = Arc::new(Mutex::new(&mut woz_track));
    let process_item = |item: &(_,_,_,_,_)| {
    	bar.inc(1);
        bar.set_message(format!("{}", item.1 as f32 / 4.0));
        let mut offset = item.0;
        process_flux_data(
        	data,
            &mut offset,
            item.1,
            mutex_woz_track.clone(),
            (item.2, item.3, item.4),
            capture,
            args,
        );    	
    };
    if args.disable_parallel {
        track_entry.iter().for_each(process_item);
    } else {
        track_entry.par_iter().for_each(process_item);
    }
    woz_track
}

fn process_strm_data(
    data: &[u8],
    offset: &mut usize,
    capture: Capture,
    args: &mut Args,
) -> Vec<WozTrackEntry> {
    let data_offset = 10;
    let read_data_fn = |data: &[u8], current_offset: usize, capture: &Capture, _: Option<u8>| {
        let location = data[current_offset];
        let capture_type = data[current_offset + 1];
        let length = if *capture == Capture::Strm {
            read_a2r_u32(data, current_offset + 2)
        } else {
            read_a2r_big_u32(data, current_offset + 2)
        };
        let loop_point = if *capture == Capture::Strm {
            read_a2r_u32(data, current_offset + 6)
        } else {
            read_a2r_big_u32(data, current_offset + 6)
        };
        (location, capture_type, length, loop_point, 10) // 10 bytes read for this header
    };

    process_flux_capture(
        (data, offset, args),
        &capture,
        0xff,
        data_offset,
        read_data_fn,
        None,
    )
}

fn process_rwcp_slvd(
    data: &[u8],
    offset: &mut usize,
    args: &mut Args,
    hard_sector_count: u8,
    rwcp: bool,
) -> Vec<WozTrackEntry> {
    let capture_type_enum = if rwcp { Capture::Rwcp } else { Capture::Slvd };

    let label = get_label(&capture_type_enum);
    let debug = args.debug;
    args.resolution = read_a2r_u32(data, *offset + 1);

    if args.resolution == 0 {
        args.resolution = 62500;
    }

    a2_debug!(debug, "{label}: Resolution : {} ps", args.resolution);

    let data_offset = 9 + 4 * data[*offset + 4] as usize;
    let read_data_fn =
        |data: &[u8], current_offset: usize, _capture: &Capture, hard_sector_count: Option<u8>| {
            let capture_type = data[current_offset + 1];
            let location =
                (data[current_offset + 2] as usize + data[current_offset + 3] as usize * 256) as u8;
            let index_signals_size = data[current_offset + 4] as usize;

            let mut bytes_read = 5;

            // Skip Mirror Distance Outward and Mirror Distance Inward for SLVD
            if !rwcp {
                bytes_read += 7;
            }

            let mut index_signals = vec![0_usize; index_signals_size];
            if index_signals_size > 0 {
                for item in index_signals.iter_mut().take(index_signals_size) {
                    *item = read_a2r_u32(data, current_offset + bytes_read) as usize;
                    bytes_read += 4;
                }
            }

            let length = read_a2r_u32(data, current_offset + bytes_read);
            bytes_read += 4; // for length

            let loop_point = if index_signals_size == 0 {
                length
            } else if rwcp {
                index_signals[hard_sector_count.unwrap() as usize] as u32
            } else {
                index_signals[0] as u32
            };

            (location, capture_type, length, loop_point, bytes_read)
        };

    *offset += 16;
    process_flux_capture(
        (data, offset, args),
        &capture_type_enum,
        0x58,
        data_offset,
        read_data_fn,
        Some(hard_sector_count),
    )
}

fn process_flux_data(
    data: &[u8],
    offset: &mut usize,
    length: u32,
    mutex_woz_track: Arc<Mutex<&mut Vec<WozTrackEntry>>>,
    flux_info: (u8, u8, u32),
    capture: &Capture,
    args: &Args,
) {
    let location = flux_info.0;
    let capture_type = flux_info.1;
    let tracks = args.tracks.as_ref().cloned().unwrap_or_default();

    if !args.full_tracks && !location.is_multiple_of(4) && !tracks.contains(&location) {
        *offset += length as usize;
        return;
    }

    // Only timing and xtiming supported
    if capture_type != 1 && capture_type != 3 {
        *offset += length as usize;
        return;
    }

    // Disable timing if switch disable-timing provided
    if args.disable_timing && capture_type == 1 {
        *offset += length as usize;
        return;
    }

    // Disable xtiming if switch disable-xtiming provided
    if args.disable_xtiming && capture_type == 3 {
        *offset += length as usize;
        return;
    }

    // If accuracy is 100% and if fast loop is enabled, skip the other same locations
    if args.fast_loop {
        let woz_track = mutex_woz_track.lock().unwrap();
        if woz_track[location as usize].loop_accuracy == PERFECT_ACCURACY {
            *offset += length as usize;
            return;
        }
    }

    analyze_flux_data(
        data,
        mutex_woz_track,
        flux_info,
        *offset,
        length,
        capture,
        args,
    );

    *offset += length as usize;
}

fn analyze_flux_data(
    data: &[u8],
    mutex_woz_track: Arc<Mutex<&mut Vec<WozTrackEntry>>>,
    flux_info: (u8, u8, u32),
    offset: usize,
    length: u32,
    capture: &Capture,
    args: &Args,
) {
    let (location, capture_type, loop_point) = flux_info;
    let label = get_label(capture);
    let capture_type_str = if capture_type == 1 {
        "timing"
    } else {
        "xtiming"
    };
    let debug = args.debug;
    let bit_timing = args.bit_timing;

    a2_debug!(
        debug,
        "{label}: Track: {} Type: {} Capture Len: {} LoopPoint: {}",
        location as f32 / 4.0,
        capture_type_str,
        length,
        loop_point
    );

    a2_debug!(debug, "Bit timing = {bit_timing}");

    let flux_data = &data[offset..offset + length as usize];
    let decompressed = decrunch_stream(flux_data);
    let gap = get_gap_array(&decompressed, 0);
    let cumulative_gap = get_cumulative_gap_array(&gap);
    let normalized_gap = get_normalized_gap_array(&gap, bit_timing);

    if normalized_gap.len() < 100 {
        return;
    }

    //let decompressed = decrunch_stream_flux(&normalized_gap);

    if let Some((start, end, accuracy)) = find_loop(
        &normalized_gap,
        location,
        capture_type,
        loop_point,
        args.kmp,
    ) {
        if start < end && start < cumulative_gap.len() as u32 && end < cumulative_gap.len() as u32 {
            let start = cumulative_gap[start as usize];
            let end = cumulative_gap[end as usize];

            let loop_flux_data = &decompressed[start as usize..end as usize];
            a2_debug!(
                debug,
                "{label}: Track: {} Len: {} End: {} Accuracy: {}%",
                location as f32 / 4.0,
                loop_flux_data.len(),
                end,
                accuracy as f32 / 100.0
            );

            // Update the track if previous accuracy is not 100% or current accuracy is 100%
            // 1. Prefer xtiming over timing capture type
            // 2. If it is same capture-type, prefer higher accuracy
            // 3. If accuracy = 100% is encountered, replaced with a newer one
            let mut woz_track = mutex_woz_track.lock().unwrap();
            let old_woz_track = &woz_track[location as usize];
            if capture_type > old_woz_track.capture_type
                || (capture_type == old_woz_track.capture_type
                    && accuracy >= old_woz_track.loop_accuracy
                    && !loop_flux_data.is_empty())
            {
                let woz_entry = WozTrackEntry {
                    loop_found: true,
                    flux_data: loop_flux_data.to_vec(),
                    loop_accuracy: accuracy,
                    capture_type,
                    original_flux_data: flux_data.to_vec(),
                };
                woz_track[location as usize] = woz_entry;
            }
        }
    } else if args.use_fft {
        let mut data: Vec<_> = decompressed.clone();
        data.extend(std::iter::repeat_n(0, data.len()));
        if let Ok(end) = find_loop_point_fft(
            &data,
            (loop_point - LOOP_POINT_DELTA) as usize,
            (loop_point + LOOP_POINT_DELTA) as usize,
        ) {
            let loop_flux_data = data.iter().take(end).copied().collect();

            let mut woz_track = mutex_woz_track.lock().unwrap();
            let woz_entry = WozTrackEntry {
                loop_found: false,
                flux_data: loop_flux_data,
                loop_accuracy: 0,
                capture_type,
                original_flux_data: flux_data.to_vec(),
            };
            woz_track[location as usize] = woz_entry;
        }
    } else {
        a2_debug!(
            debug,
            "{label}: {}Track: {} Loop not found. Fallback to original data if required{}",
            red_color(true),
            location as f32 / 4.0,
            reset_color(true)
        );

        let tracks = args.tracks.as_ref().cloned().unwrap_or_default();
        let mut woz_track = mutex_woz_track.lock().unwrap();
        if (args.enable_fallback || tracks.contains(&location))
            && !woz_track[location as usize].loop_found
        {
            let woz_entry = WozTrackEntry {
                loop_found: false,
                flux_data: decompressed[0..loop_point as usize].to_vec(),
                loop_accuracy: 0,
                capture_type,
                original_flux_data: flux_data.to_vec(),
            };
            woz_track[location as usize] = woz_entry;
        }

        if args.show_failed_loop {
            println!(
                "{label}: {}Track: {} ({}) Loop not found{}",
                red_color(false),
                location as f32 / 4.0,
                location,
                reset_color(false)
            );
        }
    }
}

fn find_loop(
    normalized_gap: &[u32],
    pos: u8,
    _capture_type: u8,
    _loop_point: u32,
    kmp: bool,
) -> Option<(u32, u32, u32)> {
    const OFFSET_LIMIT: usize = 256;

    if pos >= WOZ_MAX_TRACKS as u8 {
        return None;
    }

    if normalized_gap.is_empty() || normalized_gap.len() < OFFSET_LIMIT {
        return None;
    }

    let cumulative_gap = get_cumulative_gap_array(normalized_gap);
    let loop_point = (1600000_u64 * 1020484 / 1000000) as u32;
    let lower =
        cumulative_gap.partition_point(|&p| p < loop_point.saturating_sub(LOOP_POINT_DELTA));
    let upper = cumulative_gap.partition_point(|&p| p <= loop_point + LOOP_POINT_DELTA);

    for index in 0..OFFSET_LIMIT {
        let result = if kmp {
            find_loop_using_kmp_lps(index, normalized_gap, lower, upper)
        } else {
            find_loop_using_sliding_window(index, normalized_gap, lower, upper)
        };

        if result.is_some() {
            return result;
        }
    }
    None
}

fn find_loop_using_kmp_lps(
    index: usize,
    normalized_gap: &[u32],
    lower: usize,
    upper: usize,
) -> Option<(u32, u32, u32)> {
    let normalized_gap = &normalized_gap[index..];
    let lps = compute_kmp_lps(normalized_gap);
    let lps_len = lps[lps.len() - 1];
    if lps_len == 0 {
        return None;
    }
    let period = lps.len() - lps_len;
    
    if period < lower || period > upper {
        return None;
    }   

    compute_accuracy(normalized_gap, 0, period)
        .map(|accuracy| (index as u32, (index + period) as u32, accuracy))
}

fn find_loop_using_sliding_window(
    index: usize,
    normalized_gap: &[u32],
    lower: usize,
    upper: usize,
) -> Option<(u32, u32, u32)> {
    const SAMPLE_SIZE: usize = 100;

    if index + SAMPLE_SIZE > normalized_gap.len() {
        return None;
    }

    let signature = &normalized_gap[index..index + SAMPLE_SIZE];
    let mut best_match = None;
    let mut best_accuracy = 0;

    for pos in lower..upper {
        if pos + SAMPLE_SIZE > normalized_gap.len() {
            break;
        }

        if &normalized_gap[pos..pos + SAMPLE_SIZE] == signature {
            if let Some(accuracy) = compute_accuracy(normalized_gap, index, pos) {
                if accuracy == PERFECT_ACCURACY {
                    return Some((index as u32, pos as u32, accuracy));
                }

                if accuracy >= best_accuracy {
                    best_accuracy = accuracy;
                    best_match = Some((index as u32, pos as u32, accuracy));
                }
            }
        }
    }
    best_match
}

fn compute_accuracy(normalized_gap: &[u32], start: usize, end: usize) -> Option<u32> {
    let segment = &normalized_gap[start..end];

    if segment.is_empty() {
        return None;
    }

    let compare_data = &normalized_gap[end..];
    let compare_len = compare_data.len().min(segment.len());
    if compare_len == 0 {
        return None;
    }
    let matches = segment[..compare_len]
        .iter()
        .zip(compare_data[0..compare_len].iter())
        .filter(|&(a, b)| a == b)
        .count() as u32;
    Some((matches * ACCURACY_MULTIPLIER) / compare_len as u32)
}

fn decrunch_stream(flux_record: &[u8]) -> Vec<u8> {
    let total_output_size: usize = flux_record.iter().map(|&b| b as usize).sum();
    let mut v = Vec::with_capacity(total_output_size);
    let mut count: usize = 0;
    for &b in flux_record {
        count += b as usize;
        if count > 0 && b < 255 {
            v.push(1);
            v.extend(std::iter::repeat_n(0, count - 1));
            count = 0
        }
    }
    v
}

fn _decrunch_stream_flux(flux_record: &[u32]) -> Vec<u8> {
    let total_output_size: u32 = flux_record.iter().sum();
    let mut v = Vec::with_capacity(total_output_size as usize);
    for &b in flux_record {
        if b > 0 {
            v.push(1);
            v.extend(std::iter::repeat_n(0, b as usize - 1));
        }
    }
    v
}

fn decrunch_stream_woz(flux_len: usize, flux_record: &[u32], bit_timing: u8) -> Vec<u8> {
    let bit_timing = bit_timing as u32;
    let mut v = Vec::with_capacity(flux_len);
    for &flux_total in flux_record.iter() {
        let flux_total = flux_total / bit_timing;
        if flux_total > 1 {
            v.extend(std::iter::repeat_n(0, flux_total as usize - 1));
        }
        v.push(1);
    }
    v
}

fn crunch_stream(data: &[u8], step: u8) -> Vec<u8> {
    let step = step as usize;
    let mut result = Vec::new();
    let mut i = 0;
    while i < data.len() {
        i += 1;
        let mut j: usize = 1;
        while i < data.len() && data[i] == 0 {
            i += 1;
            j += 1;
        }
        j /= step;
        while j >= 255 {
            result.push(255);
            j -= 255;
        }
        result.push(j as u8);
    }
    result
}

fn crunch_stream_woz(data: &[u8]) -> Vec<u8> {
    let mut result = Vec::new();
    let mut mask = 0x80;
    let mut value = 0;
    let mut i = 0;
    while i < data.len() {
        if data[i] > 0 {
            value |= mask
        }
        mask >>= 1;
        if mask == 0 {
            mask = 0x80;
            result.push(value);
            value = 0;
        }
        i += 1;
    }

    if mask != 0x80 {
        result.push(value);
    }

    result
}

fn get_gap_array<T: PartialEq>(data: &[T], zero: T) -> Vec<u32> {
    let mut gap = Vec::with_capacity(data.len());
    let mut i = 0;
    while i < data.len() {
        i += 1;
        let mut j = 1;
        while i < data.len() && data[i] == zero {
            i += 1;
            j += 1;
        }
        gap.push(j as u32)
    }
    gap
}

fn get_cumulative_gap_array<T: Into<u32> + Copy>(gap: &[T]) -> Vec<u32> {
    let mut cumulative_sum = 0;
    gap.iter()
        .map(|&item| {
            cumulative_sum += item.into();
            cumulative_sum
        })
        .collect()
}

fn get_normalized_gap_array<T: Into<u32> + Copy>(gap: &[T], bit_timing: u8) -> Vec<u32> {
    gap.iter()
        .map(|&item| normalized_value(item.into(), bit_timing))
        .collect()
}

fn normalized_value(value: u32, bit_timing: u8) -> u32 {
    let bit_timing = bit_timing as u32;
    ((value + (bit_timing / 2)) / bit_timing) * bit_timing
}

fn normalized_track(track: &[u8], bit_timing: u8) -> Vec<u8> {
    track
        .iter()
        .map(|&item| normalized_value(item as u32, bit_timing) as u8)
        .collect()
}

fn compare_track(track: &[u8], prev_track: &[u8], bit_timing: u8) -> bool {
    let compare_len = track.len().min(prev_track.len());
    let track = &normalized_track(&track[0..compare_len], bit_timing);
    let prev_track = &normalized_track(&prev_track[0..compare_len], bit_timing);
    let mut best_mismatch = u32::MAX;
    let iter_count = std::cmp::min(16, track.len());
    for i in 0..iter_count {
        for j in 0..iter_count {
            let mismatches: u32 = track[i..]
                .iter()
                .zip(prev_track[j..].iter())
                .map(|(&a, &b)| (a != b) as u32)
                .sum();
            if mismatches < best_mismatch {
                best_mismatch = mismatches
            }
        }
    }

    (best_mismatch as f64 / compare_len as f64) < MAX_MISMATCH_RATIO
}

fn _cross_correlation_sameness_ratio_fft(arr1: &[u8], arr2: &[u8]) -> f64 {
    // Handle empty arrays
    if arr1.is_empty() || arr2.is_empty() {
        return 0.0;
    }

    let n1 = arr1.len();
    let n2 = arr2.len();

    // Determine the required padded length for linear cross-correlation
    // This should be n1 + n2 - 1. For FFT efficiency, pad to the next power of 2.
    let min_padded_len = n1 + n2 - 1;
    let fft_size = min_padded_len.next_power_of_two();

    // Convert u8 arrays to Complex<f64> and zero-pad
    let mut input1: Vec<Complex<f64>> = arr1.iter().map(|&x| Complex::new(x as f64, 0.0)).collect();
    input1.resize(fft_size, Complex::new(0.0, 0.0));

    // For cross-correlation, we need to reverse the second array for convolution
    // (x * y)[n] = sum_m x[m] * y[n - m].
    // For cross-correlation (x_corr y)[n] = sum_m x[m] * y[n + m].
    // If h[m] = y[-m], then (x * h)[n] = sum_m x[m] * h[n - m] = sum_m x[m] * y[-(n - m)] = sum_m x[m] * y[m - n].
    // This is equivalent to cross-correlation if the output is reversed and shifted.
    // A simpler way with FFTs is to take the FFT of arr1, FFT of arr2,
    // conjugate FFT(arr2), multiply, then IFFT.
    // Or, for real signals, just reverse arr2 and perform convolution
    let mut input2: Vec<Complex<f64>> = arr2.iter().map(|&x| Complex::new(x as f64, 0.0)).collect();
    input2.resize(fft_size, Complex::new(0.0, 0.0));

    // Create a planner
    let mut planner = FftPlanner::<f64>::new();

    // Create FFT plans
    let fft_forward = planner.plan_fft_forward(fft_size);
    let fft_inverse = planner.plan_fft_inverse(fft_size);

    // Perform forward FFTs
    fft_forward.process(&mut input1);
    fft_forward.process(&mut input2);

    // Point-wise multiplication in frequency domain for cross-correlation
    // Cross-correlation(A, B) <=> IFFT( FFT(A) * conj(FFT(B)) )
    let mut cross_corr_freq: Vec<Complex<f64>> = input1
        .iter()
        .zip(input2.iter())
        .map(|(&a, &b)| {
            a * b.conj() // Multiply FFT(arr1) by conjugate of FFT(arr2)
        })
        .collect();

    // Perform inverse FFT
    fft_inverse.process(&mut cross_corr_freq);

    // Extract the real part of the cross-correlation results
    // And normalize by 1/fft_size as rustfft doesn't normalize by default.
    let cross_corr: Vec<f64> = cross_corr_freq
        .iter()
        .map(|c| c.re / fft_size as f64)
        .collect();

    // Find the absolute maximum value of the cross-correlation
    let mut max_cross_corr_value = 0.0;
    for &val in cross_corr.iter() {
        if val.abs() > max_cross_corr_value {
            max_cross_corr_value = val.abs();
        }
    }

    // Calculate the L2-norm (Euclidean norm) for each input array
    let norm_arr1 = arr1.iter().map(|&x| (x as f64).powi(2)).sum::<f64>().sqrt();
    let norm_arr2 = arr2.iter().map(|&x| (x as f64).powi(2)).sum::<f64>().sqrt();

    // Handle division by zero if an array is all zeros
    if norm_arr1 == 0.0 || norm_arr2 == 0.0 {
        return 0.0;
    }

    // Divide the absolute maximum cross-correlation by the product of the norms
    max_cross_corr_value / (norm_arr1 * norm_arr2)
}

fn get_speed(data: &[u8], offset: usize) -> u8 {
    let mut max_index = 0;
    let mut max_value = 0;
    for i in 27..=35 {
        let value = data[offset..]
            .iter()
            .take(8000)
            .map(|&item| (item == i) as usize)
            .sum();
        if value > max_value {
            max_value = value;
            max_index = i;
        }
    }

    if max_index == 0 {
        max_index = 32
    }

    max_index
}

fn main() -> Result<(), AppError> {
    let mut args = Args::parse();

    let now = std::time::Instant::now();
    let dsk = std::fs::read(&args.input)?;

    if dsk.len() < 8 {
        return Err(Box::new(std::io::Error::new(
            std::io::ErrorKind::InvalidInput,
            "Invalid A2R file",
        )));
    }

    // Check for WOZ format
    let header = read_a2r_u32(&dsk, 0);

    if header != A2R_A2R3_HEADER && header != A2R_A2R2_HEADER && header != A2R_A2R1_HEADER {
        return Err(Box::new(std::io::Error::new(
            std::io::ErrorKind::InvalidInput,
            "Invalid A2R file",
        )));
    }

    let a2r1 = header == A2R_A2R1_HEADER;
    let a2r3 = header == A2R_A2R3_HEADER;

    let mut dsk_offset = if a2r1 { 16 } else { 8 };
    let mut info = Info::default();
    let mut info_detected = a2r1;
    info.version = 1;

    let data_dsk_offset = dsk_offset;
    while dsk_offset < dsk.len() {
        let chunk_id = read_a2r_u32(&dsk, dsk_offset);
        let chunk_size = if a2r1 {
            read_a2r_big_u32(&dsk, dsk_offset + 4)
        } else {
            read_a2r_u32(&dsk, dsk_offset + 4)
        };
        dsk_offset += 8;
        match chunk_id {
            // INFO
            A2R_INFO_CHUNK => {
                if !a2r1 {
                    process_info(&dsk, dsk_offset, &mut info, a2r3);
                    print!("{info}");
                    if info.disk_type == 1 {
                        info_detected = true;
                    }
                } else {
                    print!("{info}");
                    let metadata = process_meta(&dsk, dsk_offset, chunk_size)?;
                    print_meta_information(&metadata, false);
                }
                dsk_offset += chunk_size as usize;
            }
            A2R_META_CHUNK => {
                let metadata = process_meta(&dsk, dsk_offset, chunk_size)?;
                print_meta_information(&metadata, true);
                dsk_offset += chunk_size as usize;
            }
            _ => dsk_offset += chunk_size as usize,
        }
    }

    dsk_offset = data_dsk_offset;
    if info_detected {
        while dsk_offset < dsk.len() {
            let chunk_id = read_a2r_u32(&dsk, dsk_offset);
            let chunk_size = if a2r1 {
                read_a2r_big_u32(&dsk, dsk_offset + 4)
            } else {
                read_a2r_u32(&dsk, dsk_offset + 4)
            };
            dsk_offset += 8;
            match chunk_id {
                // DATA
                A2R_DATA_CHUNK if a2r1 => {
                    let mut woz_tracks =
                        process_strm_data(&dsk, &mut dsk_offset, Capture::Data, &mut args);
                    create_woz_file(&mut woz_tracks, &info, &args)?;
                }

                // STRM
                A2R_STRM_CHUNK if info_detected => {
                    let mut woz_tracks =
                        process_strm_data(&dsk, &mut dsk_offset, Capture::Strm, &mut args);
                    create_woz_file(&mut woz_tracks, &info, &args)?;
                }

                // RWCP
                A2R_RWCP_CHUNK => {
                    let mut woz_tracks = process_rwcp_slvd(
                        &dsk,
                        &mut dsk_offset,
                        &mut args,
                        info.hard_sector_count,
                        true,
                    );
                    create_woz_file(&mut woz_tracks, &info, &args)?;
                }

                // SLVD
                A2R_SLVD_CHUNK => {
                    let mut woz_tracks = process_rwcp_slvd(
                        &dsk,
                        &mut dsk_offset,
                        &mut args,
                        info.hard_sector_count,
                        false,
                    );
                    create_woz_file(&mut woz_tracks, &info, &args)?;
                }

                _ => dsk_offset += chunk_size as usize,
            }
        }
    }

    println!();
    if info.disk_type != 1 {
        println!("Only Disk Type 5.25 is supported");
    }
    println!("Elapsed time = {:?}", now.elapsed());

    Ok(())
}

fn process_meta(dsk: &[u8], offset: usize, chunk_size: u32) -> Result<String, FromUtf8Error> {
    String::from_utf8(dsk[offset..offset + chunk_size as usize].to_vec())
}

fn print_meta_information(meta: &str, metainfo: bool) {
    let label = if metainfo { "META" } else { "INFO" };
    for row_item in meta.lines() {
        let mut parts = row_item.splitn(2, '\t');
        if let (Some(key), Some(value)) = (parts.next(), parts.next())
            && !value.is_empty()
        {
            println!(
                "{}{label}{}: {:<20}: {}",
                green_color(false),
                reset_color(false),
                capitalize_first_letter(key),
                value
            );
        }
    }
}

fn capitalize_first_letter(s: &str) -> String {
    if s.is_empty() {
        return s.to_string();
    }
    s[0..1].to_uppercase() + &s[1..]
}

fn create_woz_file(
    woz_tracks: &mut [WozTrackEntry],
    woz_info: &Info,
    args: &Args,
) -> std::io::Result<()> {
    fn make_chunk(chunk_id: &str, data: &[u8]) -> Vec<u8> {
        let mut v = Vec::new();
        v.extend(chunk_id.as_bytes());
        write_woz_u32(&mut v, data.len() as u32);
        v.extend(data);
        v
    }

    if args.show_unsolved_tracks {
        for (i, item) in woz_tracks.iter().enumerate().take(WOZ_MAX_TRACKS) {
            if !woz_tracks[i].flux_data.is_empty() && woz_tracks[i].loop_accuracy != PERFECT_ACCURACY {
                println!(
                    "{}LOOP{}: Track {:5} ({:3})   : {}% accurate",
                    green_color(false),
                    reset_color(false),
                    i as f32 / 4.0,
                    i,
                    item.loop_accuracy / 100
                )
            }
        }
    }

    let path = Path::new(&args.output);
    let mut header = Vec::<u8>::new();
    header.extend("WOZ2".as_bytes());
    header.extend(&[0xff, 0xa, 0xd, 0xa]);
    header.extend(&[0, 0, 0, 0]);
    let mut info = vec![0u8; 60];
    info[0] = 3;
    info[1] = 1;
    info[2] = woz_info.write_protected as u8;
    info[3] = woz_info.synchronized as u8;
    info[4] = 0;
    info[5..(5 + 32)].copy_from_slice(args.creator.as_bytes());
    info[37] = 1;
    info[38] = 0;
    info[39] = 32;
    info[40..42].copy_from_slice(&[0x7f, 0]);
    info[42..44].copy_from_slice(&[48, 0]);

    let info_chunk = make_chunk("INFO", &info);

    let mut tmap_map_data = vec![0xff_u8; WOZ_MAX_TRACKS];
    let mut flux_map_data = vec![0xff_u8; WOZ_MAX_TRACKS];
    let mut track_index = 0;
    let mut flux_enabled = false;
    let mut processed_tracks: Vec<u8> = Vec::new();
    let mut prev_match_track: Option<(u8, usize)> = None;
    let ignore_tracks: Vec<usize> = args
        .delete_tracks
        .as_ref()
        .map(|delete_tracks| delete_tracks.iter().map(|&item| item as usize).collect())
        .unwrap_or_default();

    let bar = create_progress_bar(WOZ_MAX_TRACKS as u64, args.debug);
    for i in 0..WOZ_MAX_TRACKS {
        bar.inc(1);
        bar.set_message(format!("{}", i as f32 / 4.0));

        if ignore_tracks.contains(&i) {
            continue;
        }

        if args.compare_tracks {
            let mut found = 0xff;
            if i < woz_tracks.len() && !woz_tracks[i].flux_data.is_empty() {
                if let Some((prev_index, prev_match)) = prev_match_track {
                    let prev_track = &woz_tracks[prev_match].original_flux_data;
                    if compare_track(
                        &woz_tracks[i].original_flux_data,
                        prev_track,
                        args.bit_timing,
                    ) {
                        found = prev_index;
                        woz_tracks[i].flux_data.clear();
                    }
                }

                if found == 0xff {
                    found = processed_tracks.len() as u8;
                    processed_tracks.push(i as u8);
                    prev_match_track = Some((found, i))
                }
            }
            if args.woz {
                tmap_map_data[i] = found;
            } else {
                flux_map_data[i] = found;
                flux_enabled = true;
            }
        } else if args.woz {
            if i < woz_tracks.len() && !woz_tracks[i].flux_data.is_empty() {
                tmap_map_data[i] = track_index;
                if args.duplicate_quarter_tracks && i.is_multiple_of(4) {
                    duplicate_tracks(i, track_index, &mut tmap_map_data);
                }
                track_index += 1;
            }
        } else if i < woz_tracks.len() && !woz_tracks[i].flux_data.is_empty() {
            flux_map_data[i] = track_index;
            if args.duplicate_quarter_tracks && i.is_multiple_of(4) {
                duplicate_tracks(i, track_index, &mut flux_map_data);
            }
            track_index += 1;
            flux_enabled = true;
        }
    }

    if let Some(flux) = &args.flux {
        flux_enabled = true;

        // Convert tracks to flux
        for &item in flux {
            let track_index = tmap_map_data[item as usize];
            flux_map_data[item as usize] = track_index;
            tmap_map_data[item as usize] = 0xff;
        }
    }

    if let Some(tmap) = &args.tmap {
        // Convert tracks to tmap
        for &item in tmap {
            let track_index = flux_map_data[item as usize];
            tmap_map_data[item as usize] = track_index;
            flux_map_data[item as usize] = 0xff;
        }
    }

    let tmap_chunk = make_chunk("TMAP", &tmap_map_data);
    let flux_chunk = make_chunk("FLUX", &flux_map_data);
    let mut track_start_data = 3;
    let mut trks_data = Vec::<u8>::new();
    let mut fluxs = Vec::<u8>::new();
    let mut largest_tmap_track = 0;
    let mut largest_flux_track = 0;

    for i in 0..WOZ_MAX_TRACKS {
        if i < woz_tracks.len() && !woz_tracks[i].flux_data.is_empty() {
            let flux_data = &woz_tracks[i].flux_data;
            let mut step = (125000 / args.resolution) as u8;

            if step == 0 {
                step = 1;
            }

            let gap = get_gap_array(flux_data, 0);
            let bit_timing = args.bit_timing;
            let normalized_gap = get_normalized_gap_array(&gap, bit_timing * step);
            let tmap_data =
                decrunch_stream_woz(flux_data.len(), &normalized_gap, bit_timing * step);
            let data = if flux_map_data[i] != 0xff {
                &crunch_stream(flux_data, step)
            } else {
                &crunch_stream_woz(&tmap_data)
            };

            let track_len = data.len().div_ceil(WOZ_BLOCK_SIZE) * WOZ_BLOCK_SIZE;
            let mut track_data = vec![0u8; track_len];
            let starting_block = track_start_data;
            track_data[0..data.len()].copy_from_slice(data);
            let block_count = track_len / WOZ_BLOCK_SIZE;
            let bit_count = if flux_map_data[i] != 0xff {
                data.len()
            } else {
                tmap_data.len()
            };
            track_start_data += block_count;
            fluxs.extend(&track_data);
            let mut trk_data = Vec::new();
            write_woz_u16(&mut trk_data, starting_block as u16);
            write_woz_u16(&mut trk_data, block_count as u16);
            write_woz_u32(&mut trk_data, bit_count as u32);
            trks_data.extend(&trk_data);

            if flux_map_data[i] == 0xff && block_count > largest_tmap_track {
                largest_tmap_track = block_count
            }

            if flux_map_data[i] != 0xff && block_count > largest_flux_track {
                largest_flux_track = block_count
            }
        }
    }
    trks_data.extend(std::iter::repeat_n(0, WOZ_MAX_TRACKS * 8 - trks_data.len()));
    trks_data.extend(&fluxs);

    let trks_chunk = make_chunk("TRKS", &trks_data);

    // Final assembly
    let mut contents = Vec::new();
    contents.extend(&header);
    contents.extend(&info_chunk);

    contents.extend(&tmap_chunk);
    contents.extend(&trks_chunk);

    if !contents.len().is_multiple_of(WOZ_BLOCK_SIZE) {
        let padding = WOZ_BLOCK_SIZE - contents.len() % WOZ_BLOCK_SIZE;
        contents.extend(std::iter::repeat_n(0, padding));
    }

    contents[64] = (largest_tmap_track & 0xff) as u8;
    contents[65] = ((largest_tmap_track >> 8) & 0xff) as u8;

    if flux_enabled {
        let flux_block = contents.len() / 512;
        contents[66..68].copy_from_slice(&(flux_block as u16).to_le_bytes());
        contents[68..70].copy_from_slice(&(largest_flux_track as u16).to_le_bytes());
        contents.extend(&flux_chunk)
    } else {
        // Set the WOZ version to version 2.0
        contents[20] = 2
    }

    let crc = crc32(0, &contents[header.len()..]);
    contents[8..12].copy_from_slice(&crc.to_le_bytes());

    let mut file = std::fs::File::create(path)?;
    file.write_all(&contents)?;

    Ok(())
}

fn duplicate_tracks(index: usize, track_index: u8, map: &mut [u8]) {
    if index > 0 {
        map[index - 1] = track_index;
    }
    if index + 1 < WOZ_MAX_TRACKS {
        map[index + 1] = track_index;
    }

    if index >= 8 && index + 2 < WOZ_MAX_TRACKS {
        map[index + 2] = track_index;
    }
}

fn generate_crc32_table() -> [u32; 256] {
    let polynomial = 0xedb88320;
    let mut table = [0u32; 256];
    for (i, item) in table.iter_mut().enumerate() {
        let mut crc = i as u32;
        for _ in 0..8 {
            crc = if crc & 1 > 0 {
                (crc >> 1) ^ polynomial
            } else {
                crc >> 1
            };
        }
        *item = crc;
    }
    table
}

fn crc32(value: u32, buf: &[u8]) -> u32 {
    let mut crc = value ^ 0xffffffff;
    let crc32_table = generate_crc32_table();

    for data in buf {
        crc = crc32_table[((crc ^ *data as u32) & 0xff) as usize] ^ (crc >> 8);
    }
    crc ^ 0xffffffff
}

fn create_progress_bar(size: u64, debug: bool) -> ProgressBar {
    let bar = if debug {
        ProgressBar::hidden()
    } else {
        ProgressBar::new(size)
    };
    let progress_style = ProgressStyle::default_bar()
        .template("Processing Track:{msg:>5} {bar:35.green/black} {pos:>7}/{len:7}")
        .unwrap();
    bar.set_style(progress_style.progress_chars("\u{2588}\u{258c} "));
    bar
}

/// Finds the disk’s rotational period (loop point) using FFT-based autocorrelation.
///
/// # Arguments
///
/// * `bits` - An iterable of 0s or 1s, representing the data.
/// *  Its length must be at least `2 * max_expected`.
/// * `min_expected` - The minimum expected loop point (search window in bits).
/// * `max_expected` - The maximum expected loop point (search window in bits).
///
/// # Returns
///
/// A `Result` containing the exact bits per revolution as an `isize` on success,
/// or a `Box<dyn Error>` on failure.
///
/// # Errors
///
/// Returns an error if the input `bits` length is less than `2 * max_expected`.
fn find_loop_point_fft(
    bits: &[u8],
    min_expected: usize,
    max_expected: usize,
) -> Result<usize, Box<dyn Error>> {
    let n = bits.len();

    if n < 2 * max_expected {
        return Err("Need >= 2 revolutions worth of data".into());
    }

    // Map 0/1 to f32 and create a Complex vector for FFT
    let x: Vec<_> = bits.iter().map(|&b| Complex::new(b as f32, 0.0)).collect();

    // Create an FFT planner
    let mut planner = FftPlanner::new();
    let fft_forward = planner.plan_fft_forward(n);
    let fft_inverse = planner.plan_fft_inverse(n);

    // ---- Circular (unbiased) autocorrelation via FFT ------------------------

    // 1. Compute FFT of x
    let mut x_fft = x.clone(); // Clone to keep original x for other potential uses
    fft_forward.process(&mut x_fft);

    // 2. Compute X * X.conj()
    let mut ac_complex: Vec<Complex<f32>> = x_fft.iter().map(|&val| val * val.conj()).collect();

    // 3. Compute IFFT of (x_fft * x_fft.conj())
    fft_inverse.process(&mut ac_complex);

    // Extract the real part for autocorrelation
    let mut ac: Vec<f32> = ac_complex.iter().map(|c| c.re).collect();

    // Unbiased scaling: ac = ac / (n - np.arange(n))
    for (i, item) in ac.iter_mut().enumerate().take(n) {
        let divisor = (n - i) as f32;
        if divisor.abs() > f32::EPSILON {
            // Avoid division by zero
            *item /= divisor;
        } else {
            *item = 0.0; // Or handle as appropriate for your application
        }
    }

    // ---- Search the desired range for the highest peak ----------------------
    let mut max_val = f32::NEG_INFINITY;
    let mut peak_idx: usize = 0;

    // Ensure the slice bounds are within the vector's length
    let start_idx = min_expected;
    let end_idx = (max_expected + 1).min(n); // max_expected + 1 for inclusive end

    if start_idx >= end_idx {
        return Err("Search range is invalid or empty".into());
    }

    for (i, item) in ac.iter().enumerate().take(end_idx).skip(start_idx) {
        if *item > max_val {
            max_val = *item;
            peak_idx = i;
        }
    }

    Ok(peak_idx)
}

/// Custom value parser for the `--tracks` argument
///
/// Parses a comma-separated string of track numbers and ranges (e.g., "1,3-5,8").
/// Returns a sorted `Vec<u8>` of unique track numbers.
fn parse_track_ranges(s: &str) -> Result<Vec<u8>, Box<dyn Error + Sync + Send>> {
    let mut tracks = BTreeSet::new();
    for item in s.split(',') {
        let item = item.trim();
        if item.is_empty() {
            continue;
        }

        if let Some((start_str, end_str)) = item.split_once('-') {
            let start = u8::from_str(start_str.trim())
                .map_err(|_| format!("Invalid number in range start: '{start_str}'"))?;
            let end = u8::from_str(end_str.trim())
                .map_err(|_| format!("Invalid number in range end: '{end_str}'"))?;

            if start > end {
                return Err(
                    format!("Start of range ({start}) cannot be greater than end ({end})").into(),
                );
            }

            tracks.extend(start..=end);
        } else {
            // Handle single track numbers (e.g., "1", "8")
            let track_num =
                u8::from_str(item).map_err(|_| format!("Invalid track number: '{item}'"))?;
            tracks.insert(track_num);
        }
    }
    Ok(tracks.into_iter().collect())
}

/// Computes the Longest Prefix Suffix (LPS) array for the KMP algorithm.
///
/// The LPS array (also called "failure function" or "pi" array) stores for each position `i`
/// the length of the longest proper prefix of the substring `data[0..=i]` that is also a suffix.
///
/// # Arguments
/// * `data` - A slice of elements that implement the `PartialEq` trait
///
/// # Returns
/// A `Vec<usize>` where each element at index `i` contains the length of the longest
/// proper prefix which is also a suffix for the substring ending at `i`.
///
/// # Example
/// ```
/// let lps = compute_kmp_lps(b"ABABCABAB");
/// assert_eq!(lps, vec![0, 0, 1, 2, 0, 1, 2, 3, 4]);
/// ```
fn compute_kmp_lps<T: PartialEq>(data: &[T]) -> Vec<usize> {
    // Initialize LPS array with zeros. First element is always 0 since a single
    // character has no proper prefix and suffix (proper prefix means not the whole string).
    let mut pi = vec![0; data.len()];

    // Start processing from the second character (index 1)
    for i in 1..data.len() {
        // Start with the length of the previous LPS value
        let mut j = pi[i - 1];

        // While we have a non-zero prefix length and the current character
        // doesn't match the character at position j, fall back to the
        // previous longest prefix-suffix
        while j > 0 && data[i] != data[j] {
            j = pi[j - 1];
        }

        // If characters match, extend the prefix-suffix length by 1
        if data[i] == data[j] {
            j += 1
        }

        // Store the computed prefix-suffix length for current position
        pi[i] = j
    }
    pi
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_track_ranges() {
        // Test single track
        let result = parse_track_ranges("5").unwrap();
        assert_eq!(result, vec![5]);

        // Test multiple tracks
        let result = parse_track_ranges("1,3,5").unwrap();
        assert_eq!(result, vec![1, 3, 5]);

        // Test range
        let result = parse_track_ranges("3-7").unwrap();
        assert_eq!(result, vec![3, 4, 5, 6, 7]);

        // Test mixed
        let result = parse_track_ranges("1,3-5,8").unwrap();
        assert_eq!(result, vec![1, 3, 4, 5, 8]);

        // Test invalid range
        let result = parse_track_ranges("5-3");
        assert!(result.is_err());

        // Test invalid number
        let result = parse_track_ranges("abc");
        assert!(result.is_err());
    }

    #[test]
    fn test_read_a2r_u32() {
        let data = [0x78, 0x56, 0x34, 0x12];
        assert_eq!(read_a2r_u32(&data, 0), 0x12345678);
    }

    #[test]
    fn test_read_a2r_big_u32() {
        let data = [0x12, 0x34, 0x56, 0x78];
        assert_eq!(read_a2r_big_u32(&data, 0), 0x12345678);
    }

    #[test]
    fn test_write_woz_u32() {
        let mut buffer = Vec::new();
        write_woz_u32(&mut buffer, 0x12345678);
        assert_eq!(buffer, vec![0x78, 0x56, 0x34, 0x12]);
    }

    #[test]
    fn test_write_woz_u16() {
        let mut buffer = Vec::new();
        write_woz_u16(&mut buffer, 0x1234);
        assert_eq!(buffer, vec![0x34, 0x12]);
    }

    #[test]
    fn test_decrunch_stream() {
        // Test simple case
        let input = vec![2, 3, 1];
        let expected = vec![1, 0, 1, 0, 0, 1];
        assert_eq!(decrunch_stream(&input), expected);

        // Test with 255 values
        let input = vec![255, 1, 255, 2];
        let mut expected = Vec::new();
        expected.push(1);
        expected.extend(std::iter::repeat_n(0, 255));
        expected.push(1);
        expected.extend(std::iter::repeat_n(0, 256));
        assert_eq!(decrunch_stream(&input), expected);
    }

    #[test]
    fn test_get_gap_array() {
        let data = vec![1, 0, 0, 1, 0, 1];
        let expected = vec![3, 2, 1];
        assert_eq!(get_gap_array(&data, 0), expected);
    }

    #[test]
    fn test_get_cumulative_gap_array() {
        let gap: Vec<u32> = vec![1, 2, 3];
        let expected = vec![1, 3, 6];
        assert_eq!(get_cumulative_gap_array(&gap), expected);
    }

    #[test]
    fn test_get_normalized_gap_array() {
        let gap: Vec<u32> = vec![3, 7, 12];
        let bit_timing = 4;
        let expected = vec![4, 8, 12];
        assert_eq!(get_normalized_gap_array(&gap, bit_timing), expected);
    }

    #[test]
    fn test_normalized_value() {
        assert_eq!(normalized_value(3, 4), 4);
        assert_eq!(normalized_value(5, 4), 4);
        assert_eq!(normalized_value(6, 4), 8);
    }

    #[test]
    fn test_compare_track() {
        let track1 = vec![1, 0, 1, 0, 1];
        let track2 = vec![1, 0, 1, 0, 1];
        assert!(compare_track(&track1, &track2, 32));

        let track3 = vec![1, 0, 0, 0, 1];
        assert!(compare_track(&track1, &track3, 32));
    }

    #[test]
    fn test_crc32() {
        // Test known CRC32 value
        let data = b"123456789";
        assert_eq!(crc32(0, data), 0xCBF43926);
    }

    #[test]
    fn test_find_loop() {
        // Create a pattern that repeats
        let mut normalized_gap = vec![4, 8, 12, 5, 8, 12, 4, 8, 12];
        for _ in 0..30 {
            normalized_gap.extend([4, 8, 12, 5, 8, 12, 4, 8, 12]);
        }
        let pos = 0;
        let capture_type = 1;
        let loop_point = 6; // Approximate loop point

        let result = find_loop(&normalized_gap, pos, capture_type, loop_point, false);
        assert!(result.is_some(), "Loop should be detected");
        let (start, end, accuracy) = result.unwrap();
        assert_eq!(start, 0);
        assert_eq!(end, 9);
        assert_eq!(accuracy, PERFECT_ACCURACY);
    }

    #[test]
    fn test_find_loop_point_fft() {
        // Create a repeating pattern
        let pattern = vec![1, 0, 1, 0, 1, 0];
        let mut bits = pattern.clone();
        bits.extend(pattern.clone()); // Two revolutions
        bits.extend(pattern.clone()); // Three revolutions

        let min_expected = 2;
        let max_expected = 6;

        let result = find_loop_point_fft(&bits, min_expected, max_expected);
        assert!(result.is_ok());
        let loop_point = result.unwrap();
        assert_eq!(loop_point, 6); // Pattern length
    }

    #[test]
    fn test_crunch_stream() {
        let data = vec![1, 0, 0, 1, 0, 1];
        let step = 1;
        let result = crunch_stream(&data, step);
        assert_eq!(result, vec![3, 2, 1]);

        // Test with step > 1
        let result = crunch_stream(&data, 2);
        assert_eq!(result, vec![1, 1, 0]);
    }

    #[test]
    fn test_crunch_stream_woz() {
        let data = vec![1, 0, 1, 0, 1, 0];
        let result = crunch_stream_woz(&data);
        assert_eq!(result, vec![0b10101000]);
    }

    #[test]
    fn test_process_meta() {
        let data = b"Key1\tValue1\nKey2\tValue2";
        let result = process_meta(data, 0, data.len() as u32);
        assert_eq!(result, Ok("Key1\tValue1\nKey2\tValue2".into()));
    }

    #[test]
    fn test_capitalize_first_letter() {
        assert_eq!(capitalize_first_letter("hello"), "Hello");
        assert_eq!(capitalize_first_letter("hELLO"), "HELLO");
        assert_eq!(capitalize_first_letter(""), "");
    }

    #[test]
    fn test_duplicate_tracks() {
        let mut map = [0xFF; 160];
        duplicate_tracks(4, 5, &mut map);

        assert_eq!(map[3], 5); // index - 1
        assert_eq!(map[5], 5); // index + 1
        assert_eq!(map[6], 0xff); // index + 2
        //
        duplicate_tracks(8, 5, &mut map);
        assert_eq!(map[7], 5); // index - 1
        assert_eq!(map[9], 5); // index + 1
        assert_eq!(map[10], 5); // index + 2
    }

    #[test]
    fn test_get_speed() {
        // Create data with a pattern that should detect bit timing 32
        let mut data = vec![0; 8000];
        for i in 0..8000 {
            if i % 32 == 0 {
                data[i] = 32;
            }
        }

        assert_eq!(get_speed(&data, 0), 32);
    }

    #[test]
    fn test_create_progress_bar() {
        let bar = create_progress_bar(100, false);
        assert_eq!(bar.length(), Some(100));
    }

    #[test]
    fn test_info_display() {
        let mut info = Info::default();
        info.version = 2;
        info.creator = "TestCreator".to_string();
        info.disk_type = 1;
        info.write_protected = true;
        info.synchronized = false;

        let display = format!("{}", info);

        assert!(display.contains("Version"));
        assert!(display.contains("2 (A2R2)"));
        assert!(display.contains("5.25-inch (140K)"));
        assert!(display.contains("Write protected     : true"));
        assert!(display.contains("Tracks synchronized : false"));
    }
}
