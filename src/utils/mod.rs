pub mod config;

pub use config::Config;

/// Common utility functions for the AI toolset (inspired by Insilico Medicine research)
pub mod common {
    use std::path::Path;
    use tokio::fs;
    use anyhow::Result;
    
    /// Check if a file exists
    pub async fn file_exists(path: &str) -> bool {
        Path::new(path).exists()
    }
    
    /// Create directory if it doesn't exist
    pub async fn ensure_dir(path: &str) -> Result<()> {
        if !Path::new(path).exists() {
            fs::create_dir_all(path).await?;
        }
        Ok(())
    }
    
    /// Get file extension
    pub fn get_file_extension(path: &str) -> Option<&str> {
        Path::new(path).extension()?.to_str()
    }
    
    /// Get file name without extension
    pub fn get_file_name_without_extension(path: &str) -> Option<&str> {
        Path::new(path).file_stem()?.to_str()
    }
    
    /// Get file size in bytes
    pub async fn get_file_size(path: &str) -> Result<u64> {
        let metadata = fs::metadata(path).await?;
        Ok(metadata.len())
    }
    
    /// Format file size in human-readable format
    pub fn format_file_size(bytes: u64) -> String {
        const UNITS: [&str; 4] = ["B", "KB", "MB", "GB"];
        let mut size = bytes as f64;
        let mut unit_index = 0;
        
        while size >= 1024.0 && unit_index < UNITS.len() - 1 {
            size /= 1024.0;
            unit_index += 1;
        }
        
        format!("{:.1} {}", size, UNITS[unit_index])
    }
    
    /// Generate a unique filename
    pub fn generate_unique_filename(base_name: &str, extension: &str) -> String {
        use std::time::{SystemTime, UNIX_EPOCH};
        let timestamp = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_millis();
        
        format!("{}_{}.{}", base_name, timestamp, extension)
    }
    
    /// Validate file path
    pub fn validate_file_path(path: &str) -> Result<()> {
        let path_obj = Path::new(path);
        
        if path_obj.has_root() {
            return Err(anyhow::anyhow!("Absolute paths are not allowed"));
        }
        
        if path_obj.to_string_lossy().contains("..") {
            return Err(anyhow::anyhow!("Path traversal is not allowed"));
        }
        
        Ok(())
    }
    
    /// Sanitize filename
    pub fn sanitize_filename(filename: &str) -> String {
        use regex::Regex;
        
        // Remove or replace invalid characters
        let re = Regex::new(r#"[<>:"/\\|?*]"#).unwrap();
        let sanitized = re.replace_all(filename, "_");
        
        // Remove leading/trailing spaces and dots
        sanitized.trim_matches(|c| c == ' ' || c == '.').to_string()
    }
    
    /// Check if path is a directory
    pub async fn is_directory(path: &str) -> bool {
        if let Ok(metadata) = fs::metadata(path).await {
            metadata.is_dir()
        } else {
            false
        }
    }
    
    /// Check if path is a file
    pub async fn is_file(path: &str) -> bool {
        if let Ok(metadata) = fs::metadata(path).await {
            metadata.is_file()
        } else {
            false
        }
    }
    
    /// List files in directory
    pub async fn list_files(dir_path: &str) -> Result<Vec<String>> {
        let mut files = Vec::new();
        let mut entries = fs::read_dir(dir_path).await?;
        
        while let Some(entry) = entries.next_entry().await? {
            let path = entry.path();
            if path.is_file() {
                if let Some(filename) = path.file_name() {
                    if let Some(filename_str) = filename.to_str() {
                        files.push(filename_str.to_string());
                    }
                }
            }
        }
        
        Ok(files)
    }
    
    /// List directories in directory
    pub async fn list_directories(dir_path: &str) -> Result<Vec<String>> {
        let mut dirs = Vec::new();
        let mut entries = fs::read_dir(dir_path).await?;
        
        while let Some(entry) = entries.next_entry().await? {
            let path = entry.path();
            if path.is_dir() {
                if let Some(dirname) = path.file_name() {
                    if let Some(dirname_str) = dirname.to_str() {
                        dirs.push(dirname_str.to_string());
                    }
                }
            }
        }
        
        Ok(dirs)
    }
    
    /// Copy file
    pub async fn copy_file(src: &str, dst: &str) -> Result<()> {
        fs::copy(src, dst).await?;
        Ok(())
    }
    
    /// Move file
    pub async fn move_file(src: &str, dst: &str) -> Result<()> {
        fs::rename(src, dst).await?;
        Ok(())
    }
    
    /// Delete file
    pub async fn delete_file(path: &str) -> Result<()> {
        fs::remove_file(path).await?;
        Ok(())
    }
    
    /// Delete directory and contents
    pub async fn delete_directory(path: &str) -> Result<()> {
        fs::remove_dir_all(path).await?;
        Ok(())
    }
    
    /// Get current working directory
    pub async fn get_current_dir() -> Result<String> {
        let current_dir = std::env::current_dir()?;
        Ok(current_dir.to_string_lossy().to_string())
    }
    
    /// Change working directory
    pub fn change_dir(path: &str) -> Result<()> {
        std::env::set_current_dir(path)?;
        Ok(())
    }
    
    /// Get file modification time
    pub async fn get_file_modification_time(path: &str) -> Result<u64> {
        let metadata = fs::metadata(path).await?;
        let modified = metadata.modified()?;
        let duration = modified.duration_since(std::time::UNIX_EPOCH)?;
        Ok(duration.as_secs())
    }
    
    /// Format timestamp
    pub fn format_timestamp(timestamp: u64) -> String {
        use chrono::{DateTime, Utc};
        let dt = DateTime::from_timestamp(timestamp as i64, 0)
            .unwrap_or_default();
        dt.format("%Y-%m-%d %H:%M:%S").to_string()
    }
    
    /// Sleep for specified duration
    pub async fn sleep(duration_ms: u64) {
        tokio::time::sleep(tokio::time::Duration::from_millis(duration_ms)).await;
    }
    
    /// Retry operation with exponential backoff
    pub async fn retry_with_backoff<F, T, E>(
        mut operation: F,
        max_retries: usize,
        initial_delay_ms: u64,
    ) -> Result<T, E>
    where
        F: FnMut() -> Result<T, E>,
        E: std::fmt::Debug,
    {
        let mut delay_ms = initial_delay_ms;
        
        for attempt in 0..=max_retries {
            match operation() {
                Ok(result) => return Ok(result),
                Err(e) => {
                    if attempt == max_retries {
                        return Err(e);
                    }
                    
                    tracing::warn!("Operation failed on attempt {}, retrying in {}ms: {:?}", 
                                  attempt + 1, delay_ms, e);
                    
                    sleep(delay_ms).await;
                    delay_ms *= 2; // Exponential backoff
                }
            }
        }
        
        unreachable!()
    }
    
    /// Generate random string
    pub fn generate_random_string(length: usize) -> String {
        use rand::Rng;
        use rand::distributions::Alphanumeric;
        
        let mut rng = rand::thread_rng();
        (0..length)
            .map(|_| rng.sample(Alphanumeric) as char)
            .collect()
    }
    
    /// Generate random integer in range
    pub fn generate_random_int(min: i64, max: i64) -> i64 {
        use rand::Rng;
        let mut rng = rand::thread_rng();
        rng.gen_range(min..=max)
    }
    
    /// Generate random float in range
    pub fn generate_random_float(min: f64, max: f64) -> f64 {
        use rand::Rng;
        let mut rng = rand::thread_rng();
        rng.gen_range(min..=max)
    }
}
