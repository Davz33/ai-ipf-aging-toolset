use serde::{Serialize, Deserialize};
use std::collections::HashMap;
use std::path::PathBuf;
use tokio::fs;
use anyhow::Result;

/// Configuration for the AI toolset (inspired by Insilico Medicine research)
/// Based on the requirements described in the research paper
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Config {
    /// Application settings
    pub app: AppConfig,
    
    /// Model configuration
    pub models: ModelsConfig,
    
    /// Data configuration
    pub data: DataConfig,
    
    /// Training configuration
    pub training: TrainingConfig,
    
    /// Output configuration
    pub output: OutputConfig,
    
    /// Logging configuration
    pub logging: LoggingConfig,
    
    /// Performance configuration
    pub performance: PerformanceConfig,
    
    /// Security configuration
    pub security: SecurityConfig,
}

/// Application configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AppConfig {
    /// Application name
    pub name: String,
    
    /// Application version
    pub version: String,
    
    /// Application description
    pub description: String,
    
    /// Author information
    pub author: String,
    
    /// License information
    pub license: String,
    
    /// Working directory
    pub working_dir: PathBuf,
    
    /// Temporary directory
    pub temp_dir: PathBuf,
    
    /// Configuration file path
    pub config_file: Option<PathBuf>,
    
    /// Environment
    pub environment: Environment,
}

/// Environment type
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum Environment {
    Development,
    Testing,
    Staging,
    Production,
}

/// Models configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelsConfig {
    /// Aging clock model configuration
    pub aging_clock: AgingClockConfig,
    
    /// IPF P3GPT model configuration
    pub ipf_p3gpt: IpfP3GptConfig,
    
    /// Transformer model configuration
    pub transformer: TransformerConfig,
    
    /// Neural network configuration
    pub neural_network: NeuralNetworkConfig,
    
    /// Model storage directory
    pub model_dir: PathBuf,
    
    /// Pre-trained model paths
    pub pretrained_models: HashMap<String, PathBuf>,
    
    /// Model validation settings
    pub validation: ModelValidationConfig,
}

/// Aging clock model configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AgingClockConfig {
    /// Model architecture
    pub architecture: String,
    
    /// Input dimension
    pub input_dim: usize,
    
    /// Hidden layer dimensions
    pub hidden_dims: Vec<usize>,
    
    /// Output dimension
    pub output_dim: usize,
    
    /// Activation function
    pub activation: String,
    
    /// Dropout rate
    pub dropout_rate: f64,
    
    /// Learning rate
    pub learning_rate: f64,
    
    /// Batch size
    pub batch_size: usize,
    
    /// Maximum epochs
    pub max_epochs: usize,
    
    /// Early stopping patience
    pub early_stopping_patience: usize,
    
    /// L2 regularization
    pub l2_reg: f64,
    
    /// Feature selection threshold
    pub feature_selection_threshold: f64,
    
    /// Cross-validation folds
    pub cv_folds: usize,
}

/// IPF P3GPT model configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IpfP3GptConfig {
    /// Model architecture
    pub architecture: String,
    
    /// Embedding dimension
    pub embedding_dim: usize,
    
    /// Number of transformer blocks
    pub num_blocks: usize,
    
    /// Number of attention heads
    pub num_heads: usize,
    
    /// Maximum sequence length
    pub max_seq_len: usize,
    
    /// Dropout rate
    pub dropout_rate: f64,
    
    /// Learning rate
    pub learning_rate: f64,
    
    /// Batch size
    pub batch_size: usize,
    
    /// Maximum epochs
    pub max_epochs: usize,
    
    /// Warmup steps
    pub warmup_steps: usize,
    
    /// Weight decay
    pub weight_decay: f64,
    
    /// Vocabulary size
    pub vocab_size: usize,
    
    /// Special tokens
    pub special_tokens: Vec<String>,
}

/// Transformer model configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TransformerConfig {
    /// Model architecture
    pub architecture: String,
    
    /// Embedding dimension
    pub embedding_dim: usize,
    
    /// Number of transformer blocks
    pub num_blocks: usize,
    
    /// Number of attention heads
    pub num_heads: usize,
    
    /// Maximum sequence length
    pub max_seq_len: usize,
    
    /// Dropout rate
    pub dropout_rate: f64,
    
    /// Learning rate
    pub learning_rate: f64,
    
    /// Batch size
    pub batch_size: usize,
    
    /// Maximum epochs
    pub max_epochs: usize,
    
    /// Warmup steps
    pub warmup_steps: usize,
    
    /// Weight decay
    pub weight_decay: f64,
    
    /// Position encoding
    pub position_encoding: String,
    
    /// Layer normalization epsilon
    pub layer_norm_epsilon: f64,
}

/// Neural network configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NeuralNetworkConfig {
    /// Model architecture
    pub architecture: String,
    
    /// Input dimension
    pub input_dim: usize,
    
    /// Hidden layer dimensions
    pub hidden_dims: Vec<usize>,
    
    /// Output dimension
    pub output_dim: usize,
    
    /// Activation function
    pub activation: String,
    
    /// Output activation function
    pub output_activation: String,
    
    /// Dropout rate
    pub dropout_rate: f64,
    
    /// Learning rate
    pub learning_rate: f64,
    
    /// Batch size
    pub batch_size: usize,
    
    /// Maximum epochs
    pub max_epochs: usize,
    
    /// Early stopping patience
    pub early_stopping_patience: usize,
    
    /// L2 regularization
    pub l2_reg: f64,
    
    /// Batch normalization
    pub batch_normalization: bool,
    
    /// Weight initialization
    pub weight_initialization: String,
}

/// Model validation configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelValidationConfig {
    /// Validation split ratio
    pub validation_split: f64,
    
    /// Test split ratio
    pub test_split: f64,
    
    /// Cross-validation folds
    pub cv_folds: usize,
    
    /// Stratified sampling
    pub stratified_sampling: bool,
    
    /// Random seed
    pub random_seed: Option<u64>,
    
    /// Metrics to track
    pub metrics: Vec<String>,
    
    /// Early stopping criteria
    pub early_stopping_criteria: Vec<String>,
}

/// Data configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DataConfig {
    /// UK Biobank configuration
    pub uk_biobank: UkBiobankConfig,
    
    /// Data preprocessing configuration
    pub preprocessing: PreprocessingConfig,
    
    /// Data validation configuration
    pub validation: DataValidationConfig,
    
    /// Data storage configuration
    pub storage: DataStorageConfig,
    
    /// Data sources
    pub sources: HashMap<String, DataSourceConfig>,
}

/// UK Biobank configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UkBiobankConfig {
    /// Data directory
    pub data_dir: PathBuf,
    
    /// Proteomics data path
    pub proteomics_path: PathBuf,
    
    /// Clinical data path
    pub clinical_path: PathBuf,
    
    /// Sample metadata path
    pub sample_metadata_path: PathBuf,
    
    /// Data access credentials
    pub credentials: Option<BiobankCredentials>,
    
    /// Data version
    pub data_version: String,
    
    /// Quality control settings
    pub quality_control: QualityControlConfig,
}

/// Biobank credentials
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BiobankCredentials {
    /// Username
    pub username: String,
    
    /// Password (encrypted)
    pub password: String,
    
    /// API key
    pub api_key: Option<String>,
    
    /// Access token
    pub access_token: Option<String>,
}

/// Quality control configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QualityControlConfig {
    /// Missing data threshold
    pub missing_data_threshold: f64,
    
    /// Outlier detection method
    pub outlier_detection_method: String,
    
    /// Outlier threshold
    pub outlier_threshold: f64,
    
    /// Coefficient of variation threshold
    pub cv_threshold: f64,
    
    /// Signal-to-noise ratio threshold
    pub snr_threshold: f64,
    
    /// Detection rate threshold
    pub detection_rate_threshold: f64,
}

/// Data preprocessing configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PreprocessingConfig {
    /// Missing value handling strategy
    pub missing_value_strategy: String,
    
    /// Feature scaling strategy
    pub feature_scaling_strategy: String,
    
    /// Outlier handling strategy
    pub outlier_handling_strategy: String,
    
    /// Feature selection strategy
    pub feature_selection_strategy: String,
    
    /// Data augmentation strategy
    pub data_augmentation_strategy: String,
    
    /// Normalization method
    pub normalization_method: String,
    
    /// Categorical encoding method
    pub categorical_encoding_method: String,
    
    /// Feature interaction generation
    pub feature_interaction_generation: bool,
    
    /// Polynomial feature degree
    pub polynomial_feature_degree: usize,
}

/// Data validation configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DataValidationConfig {
    /// Schema validation
    pub schema_validation: bool,
    
    /// Data type validation
    pub data_type_validation: bool,
    
    /// Range validation
    pub range_validation: bool,
    
    /// Consistency validation
    pub consistency_validation: bool,
    
    /// Completeness validation
    pub completeness_validation: bool,
    
    /// Uniqueness validation
    pub uniqueness_validation: bool,
    
    /// Format validation
    pub format_validation: bool,
    
    /// Business rule validation
    pub business_rule_validation: bool,
}

/// Data storage configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DataStorageConfig {
    /// Storage type
    pub storage_type: String,
    
    /// Local storage path
    pub local_path: Option<PathBuf>,
    
    /// Database connection string
    pub database_url: Option<String>,
    
    /// Cloud storage configuration
    pub cloud_storage: Option<CloudStorageConfig>,
    
    /// Compression settings
    pub compression: CompressionConfig,
    
    /// Encryption settings
    pub encryption: EncryptionConfig,
    
    /// Backup settings
    pub backup: BackupConfig,
}

/// Cloud storage configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CloudStorageConfig {
    /// Provider (AWS, GCP, Azure)
    pub provider: String,
    
    /// Bucket name
    pub bucket_name: String,
    
    /// Region
    pub region: String,
    
    /// Access key ID
    pub access_key_id: String,
    
    /// Secret access key
    pub secret_access_key: String,
    
    /// Endpoint URL
    pub endpoint_url: Option<String>,
}

/// Compression configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CompressionConfig {
    /// Compression algorithm
    pub algorithm: String,
    
    /// Compression level
    pub level: u8,
    
    /// Compress on write
    pub compress_on_write: bool,
    
    /// Decompress on read
    pub decompress_on_read: bool,
}

/// Encryption configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EncryptionConfig {
    /// Encryption algorithm
    pub algorithm: String,
    
    /// Key size
    pub key_size: usize,
    
    /// Encryption key
    pub encryption_key: Option<String>,
    
    /// Encrypt at rest
    pub encrypt_at_rest: bool,
    
    /// Encrypt in transit
    pub encrypt_in_transit: bool,
}

/// Backup configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BackupConfig {
    /// Backup frequency
    pub frequency: String,
    
    /// Backup retention
    pub retention: String,
    
    /// Backup location
    pub location: PathBuf,
    
    /// Incremental backup
    pub incremental: bool,
    
    /// Compression
    pub compression: bool,
    
    /// Encryption
    pub encryption: bool,
}

/// Data source configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DataSourceConfig {
    /// Source type
    pub source_type: String,
    
    /// Source path
    pub source_path: PathBuf,
    
    /// Format
    pub format: String,
    
    /// Delimiter (for CSV)
    pub delimiter: Option<char>,
    
    /// Has header
    pub has_header: bool,
    
    /// Encoding
    pub encoding: String,
    
    /// Compression
    pub compression: Option<String>,
}

/// Training configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrainingConfig {
    /// Training mode
    pub mode: TrainingMode,
    
    /// Distributed training configuration
    pub distributed: DistributedTrainingConfig,
    
    /// Hyperparameter optimization
    pub hyperparameter_optimization: HyperparameterOptimizationConfig,
    
    /// Model checkpointing
    pub checkpointing: CheckpointingConfig,
    
    /// Experiment tracking
    pub experiment_tracking: ExperimentTrackingConfig,
}

/// Training mode
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum TrainingMode {
    SingleGPU,
    MultiGPU,
    Distributed,
    Cloud,
}

/// Distributed training configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DistributedTrainingConfig {
    /// Number of nodes
    pub num_nodes: usize,
    
    /// Number of GPUs per node
    pub gpus_per_node: usize,
    
    /// Master address
    pub master_addr: String,
    
    /// Master port
    pub master_port: u16,
    
    /// Backend
    pub backend: String,
    
    /// Communication algorithm
    pub communication_algorithm: String,
}

/// Hyperparameter optimization configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HyperparameterOptimizationConfig {
    /// Optimization method
    pub method: String,
    
    /// Number of trials
    pub num_trials: usize,
    
    /// Search space
    pub search_space: HashMap<String, String>,
    
    /// Optimization metric
    pub optimization_metric: String,
    
    /// Direction (minimize/maximize)
    pub direction: String,
    
    /// Pruning
    pub pruning: bool,
}

/// Checkpointing configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CheckpointingConfig {
    /// Checkpoint frequency
    pub frequency: usize,
    
    /// Checkpoint directory
    pub directory: PathBuf,
    
    /// Keep last N checkpoints
    pub keep_last_n: usize,
    
    /// Save best model
    pub save_best: bool,
    
    /// Save optimizer state
    pub save_optimizer: bool,
    
    /// Save scheduler state
    pub save_scheduler: bool,
}

/// Experiment tracking configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExperimentTrackingConfig {
    /// Tracking backend
    pub backend: String,
    
    /// Project name
    pub project_name: String,
    
    /// Experiment name
    pub experiment_name: String,
    
    /// Tags
    pub tags: Vec<String>,
    
    /// Parameters to track
    pub parameters: Vec<String>,
    
    /// Metrics to track
    pub metrics: Vec<String>,
    
    /// Artifacts to track
    pub artifacts: Vec<String>,
}

/// Output configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OutputConfig {
    /// Output directory
    pub output_dir: PathBuf,
    
    /// Results directory
    pub results_dir: PathBuf,
    
    /// Logs directory
    pub logs_dir: PathBuf,
    
    /// Reports directory
    pub reports_dir: PathBuf,
    
    /// Visualizations directory
    pub visualizations_dir: PathBuf,
    
    /// Export formats
    pub export_formats: Vec<String>,
    
    /// Output compression
    pub compression: bool,
    
    /// Output encryption
    pub encryption: bool,
}

/// Logging configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LoggingConfig {
    /// Log level
    pub level: String,
    
    /// Log format
    pub format: String,
    
    /// Log file path
    pub file_path: Option<PathBuf>,
    
    /// Console logging
    pub console: bool,
    
    /// File logging
    pub file: bool,
    
    /// Structured logging
    pub structured: bool,
    
    /// Log rotation
    pub rotation: LogRotationConfig,
}

/// Log rotation configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LogRotationConfig {
    /// Max file size
    pub max_file_size: u64,
    
    /// Max files
    pub max_files: usize,
    
    /// Rotation interval
    pub rotation_interval: String,
}

/// Performance configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceConfig {
    /// Number of threads
    pub num_threads: usize,
    
    /// Memory limit
    pub memory_limit: Option<u64>,
    
    /// GPU memory fraction
    pub gpu_memory_fraction: f64,
    
    /// Mixed precision training
    pub mixed_precision: bool,
    
    /// Gradient accumulation
    pub gradient_accumulation: usize,
    
    /// Data loading workers
    pub data_loading_workers: usize,
    
    /// Prefetch factor
    pub prefetch_factor: usize,
}

/// Security configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SecurityConfig {
    /// Authentication required
    pub authentication_required: bool,
    
    /// Authorization required
    pub authorization_required: bool,
    
    /// SSL/TLS enabled
    pub ssl_enabled: bool,
    
    /// Certificate path
    pub certificate_path: Option<PathBuf>,
    
    /// Private key path
    pub private_key_path: Option<PathBuf>,
    
    /// Allowed origins
    pub allowed_origins: Vec<String>,
    
    /// Rate limiting
    pub rate_limiting: RateLimitingConfig,
}

/// Rate limiting configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RateLimitingConfig {
    /// Enabled
    pub enabled: bool,
    
    /// Requests per minute
    pub requests_per_minute: usize,
    
    /// Burst size
    pub burst_size: usize,
}

impl Config {
    /// Create a new configuration with default values
    pub fn new() -> Self {
        Self {
            app: AppConfig::default(),
            models: ModelsConfig::default(),
            data: DataConfig::default(),
            training: TrainingConfig::default(),
            output: OutputConfig::default(),
            logging: LoggingConfig::default(),
            performance: PerformanceConfig::default(),
            security: SecurityConfig::default(),
        }
    }
    
    /// Load configuration from file
    pub async fn from_file(path: &str) -> Result<Self> {
        let config_content = fs::read_to_string(path).await?;
        let config: Config = serde_json::from_str(&config_content)?;
        Ok(config)
    }
    
    /// Save configuration to file
    pub async fn save_to_file(&self, path: &str) -> Result<()> {
        let config_content = serde_json::to_string_pretty(self)?;
        fs::write(path, config_content).await?;
        Ok(())
    }
    
    /// Merge with another configuration
    pub fn merge(&mut self, other: &Config) {
        // This would implement deep merging logic
        // For now, just replace the entire configuration
        *self = other.clone();
    }
    
    /// Validate configuration
    pub fn validate(&self) -> Result<()> {
        // Validate required fields
        if self.app.name.is_empty() {
            return Err(anyhow::anyhow!("Application name cannot be empty"));
        }
        
        if self.app.version.is_empty() {
            return Err(anyhow::anyhow!("Application version cannot be empty"));
        }
        
        // Validate paths
        if !self.app.working_dir.exists() {
            return Err(anyhow::anyhow!("Working directory does not exist: {:?}", self.app.working_dir));
        }
        
        // Validate model configurations
        self.models.validate()?;
        
        // Validate data configurations
        self.data.validate()?;
        
        Ok(())
    }
    
    /// Get configuration value by key path
    pub fn get_value(&self, _key_path: &str) -> Option<serde_json::Value> {
        // This would implement key path resolution
        // For now, return None
        None
    }
    
    /// Set configuration value by key path
    pub fn set_value(&mut self, _key_path: &str, _value: serde_json::Value) -> Result<()> {
        // This would implement key path resolution and setting
        // For now, return an error
        Err(anyhow::anyhow!("Setting configuration values by key path not yet implemented"))
    }
}

impl Default for AppConfig {
    fn default() -> Self {
        Self {
            name: "AI Toolset for IPF and Aging Research".to_string(),
            version: "1.0.0".to_string(),
            description: "AI-driven toolset for IPF and aging research".to_string(),
            author: "Davide Vitiello".to_string(),
            license: "MIT".to_string(),
            working_dir: PathBuf::from("."),
            temp_dir: PathBuf::from("/tmp"),
            config_file: None,
            environment: Environment::Development,
        }
    }
}

impl Default for ModelsConfig {
    fn default() -> Self {
        Self {
            aging_clock: AgingClockConfig::default(),
            ipf_p3gpt: IpfP3GptConfig::default(),
            transformer: TransformerConfig::default(),
            neural_network: NeuralNetworkConfig::default(),
            model_dir: PathBuf::from("./models"),
            pretrained_models: HashMap::new(),
            validation: ModelValidationConfig::default(),
        }
    }
}

impl Default for AgingClockConfig {
    fn default() -> Self {
        Self {
            architecture: "feedforward".to_string(),
            input_dim: 1000,
            hidden_dims: vec![512, 256, 128],
            output_dim: 1,
            activation: "relu".to_string(),
            dropout_rate: 0.2,
            learning_rate: 0.001,
            batch_size: 32,
            max_epochs: 100,
            early_stopping_patience: 10,
            l2_reg: 0.01,
            feature_selection_threshold: 0.01,
            cv_folds: 5,
        }
    }
}

impl Default for IpfP3GptConfig {
    fn default() -> Self {
        Self {
            architecture: "transformer".to_string(),
            embedding_dim: 128,
            num_blocks: 6,
            num_heads: 8,
            max_seq_len: 512,
            dropout_rate: 0.1,
            learning_rate: 0.0001,
            batch_size: 16,
            max_epochs: 100,
            warmup_steps: 1000,
            weight_decay: 0.01,
            vocab_size: 10000,
            special_tokens: vec!["[PAD]".to_string(), "[UNK]".to_string(), "[CLS]".to_string(), "[SEP]".to_string()],
        }
    }
}

impl Default for TransformerConfig {
    fn default() -> Self {
        Self {
            architecture: "transformer".to_string(),
            embedding_dim: 128,
            num_blocks: 6,
            num_heads: 8,
            max_seq_len: 512,
            dropout_rate: 0.1,
            learning_rate: 0.0001,
            batch_size: 32,
            max_epochs: 100,
            warmup_steps: 1000,
            weight_decay: 0.01,
            position_encoding: "sinusoidal".to_string(),
            layer_norm_epsilon: 1e-12,
        }
    }
}

impl Default for NeuralNetworkConfig {
    fn default() -> Self {
        Self {
            architecture: "feedforward".to_string(),
            input_dim: 100,
            hidden_dims: vec![64, 32],
            output_dim: 1,
            activation: "relu".to_string(),
            output_activation: "linear".to_string(),
            dropout_rate: 0.2,
            learning_rate: 0.001,
            batch_size: 32,
            max_epochs: 100,
            early_stopping_patience: 10,
            l2_reg: 0.01,
            batch_normalization: true,
            weight_initialization: "xavier".to_string(),
        }
    }
}

impl Default for ModelValidationConfig {
    fn default() -> Self {
        Self {
            validation_split: 0.2,
            test_split: 0.2,
            cv_folds: 5,
            stratified_sampling: true,
            random_seed: Some(42),
            metrics: vec!["accuracy".to_string(), "precision".to_string(), "recall".to_string(), "f1".to_string()],
            early_stopping_criteria: vec!["validation_loss".to_string()],
        }
    }
}

impl Default for DataConfig {
    fn default() -> Self {
        Self {
            uk_biobank: UkBiobankConfig::default(),
            preprocessing: PreprocessingConfig::default(),
            validation: DataValidationConfig::default(),
            storage: DataStorageConfig::default(),
            sources: HashMap::new(),
        }
    }
}

impl Default for UkBiobankConfig {
    fn default() -> Self {
        Self {
            data_dir: PathBuf::from("./data/uk_biobank"),
            proteomics_path: PathBuf::from("./data/uk_biobank/proteomics.csv"),
            clinical_path: PathBuf::from("./data/uk_biobank/clinical.csv"),
            sample_metadata_path: PathBuf::from("./data/uk_biobank/metadata.csv"),
            credentials: None,
            data_version: "1.0".to_string(),
            quality_control: QualityControlConfig::default(),
        }
    }
}

impl Default for QualityControlConfig {
    fn default() -> Self {
        Self {
            missing_data_threshold: 0.2,
            outlier_detection_method: "iqr".to_string(),
            outlier_threshold: 3.0,
            cv_threshold: 0.3,
            snr_threshold: 10.0,
            detection_rate_threshold: 0.8,
        }
    }
}

impl Default for PreprocessingConfig {
    fn default() -> Self {
        Self {
            missing_value_strategy: "mean_imputation".to_string(),
            feature_scaling_strategy: "standard_scaler".to_string(),
            outlier_handling_strategy: "cap".to_string(),
            feature_selection_strategy: "variance_threshold".to_string(),
            data_augmentation_strategy: "none".to_string(),
            normalization_method: "z_score".to_string(),
            categorical_encoding_method: "one_hot".to_string(),
            feature_interaction_generation: false,
            polynomial_feature_degree: 2,
        }
    }
}

impl Default for DataValidationConfig {
    fn default() -> Self {
        Self {
            schema_validation: true,
            data_type_validation: true,
            range_validation: true,
            consistency_validation: true,
            completeness_validation: true,
            uniqueness_validation: true,
            format_validation: true,
            business_rule_validation: true,
        }
    }
}

impl Default for DataStorageConfig {
    fn default() -> Self {
        Self {
            storage_type: "local".to_string(),
            local_path: Some(PathBuf::from("./data")),
            database_url: None,
            cloud_storage: None,
            compression: CompressionConfig::default(),
            encryption: EncryptionConfig::default(),
            backup: BackupConfig::default(),
        }
    }
}

impl Default for CompressionConfig {
    fn default() -> Self {
        Self {
            algorithm: "gzip".to_string(),
            level: 6,
            compress_on_write: true,
            decompress_on_read: true,
        }
    }
}

impl Default for EncryptionConfig {
    fn default() -> Self {
        Self {
            algorithm: "AES-256".to_string(),
            key_size: 256,
            encryption_key: None,
            encrypt_at_rest: false,
            encrypt_in_transit: true,
        }
    }
}

impl Default for BackupConfig {
    fn default() -> Self {
        Self {
            frequency: "daily".to_string(),
            retention: "30 days".to_string(),
            location: PathBuf::from("./backups"),
            incremental: true,
            compression: true,
            encryption: false,
        }
    }
}

impl Default for DataSourceConfig {
    fn default() -> Self {
        Self {
            source_type: "file".to_string(),
            source_path: PathBuf::from("./data"),
            format: "csv".to_string(),
            delimiter: Some(','),
            has_header: true,
            encoding: "utf-8".to_string(),
            compression: None,
        }
    }
}

impl Default for TrainingConfig {
    fn default() -> Self {
        Self {
            mode: TrainingMode::SingleGPU,
            distributed: DistributedTrainingConfig::default(),
            hyperparameter_optimization: HyperparameterOptimizationConfig::default(),
            checkpointing: CheckpointingConfig::default(),
            experiment_tracking: ExperimentTrackingConfig::default(),
        }
    }
}

impl Default for DistributedTrainingConfig {
    fn default() -> Self {
        Self {
            num_nodes: 1,
            gpus_per_node: 1,
            master_addr: "localhost".to_string(),
            master_port: 29500,
            backend: "nccl".to_string(),
            communication_algorithm: "allreduce".to_string(),
        }
    }
}

impl Default for HyperparameterOptimizationConfig {
    fn default() -> Self {
        Self {
            method: "random_search".to_string(),
            num_trials: 100,
            search_space: HashMap::new(),
            optimization_metric: "validation_loss".to_string(),
            direction: "minimize".to_string(),
            pruning: false,
        }
    }
}

impl Default for CheckpointingConfig {
    fn default() -> Self {
        Self {
            frequency: 10,
            directory: PathBuf::from("./checkpoints"),
            keep_last_n: 5,
            save_best: true,
            save_optimizer: true,
            save_scheduler: true,
        }
    }
}

impl Default for ExperimentTrackingConfig {
    fn default() -> Self {
        Self {
            backend: "wandb".to_string(),
            project_name: "insilicomed".to_string(),
            experiment_name: "experiment".to_string(),
            tags: vec![],
            parameters: vec![],
            metrics: vec![],
            artifacts: vec![],
        }
    }
}

impl Default for OutputConfig {
    fn default() -> Self {
        Self {
            output_dir: PathBuf::from("./output"),
            results_dir: PathBuf::from("./output/results"),
            logs_dir: PathBuf::from("./output/logs"),
            reports_dir: PathBuf::from("./output/reports"),
            visualizations_dir: PathBuf::from("./output/visualizations"),
            export_formats: vec!["json".to_string(), "csv".to_string(), "png".to_string()],
            compression: true,
            encryption: false,
        }
    }
}

impl Default for LoggingConfig {
    fn default() -> Self {
        Self {
            level: "info".to_string(),
            format: "json".to_string(),
            file_path: Some(PathBuf::from("./logs/app.log")),
            console: true,
            file: true,
            structured: true,
            rotation: LogRotationConfig::default(),
        }
    }
}

impl Default for LogRotationConfig {
    fn default() -> Self {
        Self {
            max_file_size: 100 * 1024 * 1024, // 100 MB
            max_files: 10,
            rotation_interval: "daily".to_string(),
        }
    }
}

impl Default for PerformanceConfig {
    fn default() -> Self {
        Self {
            num_threads: std::thread::available_parallelism().map(|p| p.get()).unwrap_or(1),
            memory_limit: None,
            gpu_memory_fraction: 0.8,
            mixed_precision: false,
            gradient_accumulation: 1,
            data_loading_workers: 4,
            prefetch_factor: 2,
        }
    }
}

impl Default for SecurityConfig {
    fn default() -> Self {
        Self {
            authentication_required: false,
            authorization_required: false,
            ssl_enabled: false,
            certificate_path: None,
            private_key_path: None,
            allowed_origins: vec!["*".to_string()],
            rate_limiting: RateLimitingConfig::default(),
        }
    }
}

impl Default for RateLimitingConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            requests_per_minute: 1000,
            burst_size: 100,
        }
    }
}

impl ModelsConfig {
    fn validate(&self) -> Result<()> {
        // Validate model configurations
        self.aging_clock.validate()?;
        self.ipf_p3gpt.validate()?;
        self.transformer.validate()?;
        self.neural_network.validate()?;
        Ok(())
    }
}

impl AgingClockConfig {
    fn validate(&self) -> Result<()> {
        if self.input_dim == 0 {
            return Err(anyhow::anyhow!("Input dimension must be greater than 0"));
        }
        
        if self.output_dim == 0 {
            return Err(anyhow::anyhow!("Output dimension must be greater than 0"));
        }
        
        if self.learning_rate <= 0.0 {
            return Err(anyhow::anyhow!("Learning rate must be greater than 0"));
        }
        
        if self.dropout_rate < 0.0 || self.dropout_rate >= 1.0 {
            return Err(anyhow::anyhow!("Dropout rate must be between 0 and 1"));
        }
        
        Ok(())
    }
}

impl IpfP3GptConfig {
    fn validate(&self) -> Result<()> {
        if self.embedding_dim == 0 {
            return Err(anyhow::anyhow!("Embedding dimension must be greater than 0"));
        }
        
        if self.num_blocks == 0 {
            return Err(anyhow::anyhow!("Number of blocks must be greater than 0"));
        }
        
        if self.num_heads == 0 {
            return Err(anyhow::anyhow!("Number of heads must be greater than 0"));
        }
        
        if self.embedding_dim % self.num_heads != 0 {
            return Err(anyhow::anyhow!("Embedding dimension must be divisible by number of heads"));
        }
        
        Ok(())
    }
}

impl TransformerConfig {
    fn validate(&self) -> Result<()> {
        if self.embedding_dim == 0 {
            return Err(anyhow::anyhow!("Embedding dimension must be greater than 0"));
        }
        
        if self.num_blocks == 0 {
            return Err(anyhow::anyhow!("Number of blocks must be greater than 0"));
        }
        
        if self.num_heads == 0 {
            return Err(anyhow::anyhow!("Number of heads must be greater than 0"));
        }
        
        if self.embedding_dim % self.num_heads != 0 {
            return Err(anyhow::anyhow!("Embedding dimension must be divisible by number of heads"));
        }
        
        Ok(())
    }
}

impl NeuralNetworkConfig {
    fn validate(&self) -> Result<()> {
        if self.input_dim == 0 {
            return Err(anyhow::anyhow!("Input dimension must be greater than 0"));
        }
        
        if self.output_dim == 0 {
            return Err(anyhow::anyhow!("Output dimension must be greater than 0"));
        }
        
        if self.learning_rate <= 0.0 {
            return Err(anyhow::anyhow!("Learning rate must be greater than 0"));
        }
        
        if self.dropout_rate < 0.0 || self.dropout_rate >= 1.0 {
            return Err(anyhow::anyhow!("Dropout rate must be between 0 and 1"));
        }
        
        Ok(())
    }
}

impl DataConfig {
    fn validate(&self) -> Result<()> {
        // Validate UK Biobank configuration
        self.uk_biobank.validate()?;
        
        // Validate preprocessing configuration
        self.preprocessing.validate()?;
        
        // Validate storage configuration
        self.storage.validate()?;
        
        Ok(())
    }
}

impl UkBiobankConfig {
    fn validate(&self) -> Result<()> {
        if self.data_dir.to_string_lossy().is_empty() {
            return Err(anyhow::anyhow!("Data directory cannot be empty"));
        }
        
        if self.proteomics_path.to_string_lossy().is_empty() {
            return Err(anyhow::anyhow!("Proteomics path cannot be empty"));
        }
        
        if self.clinical_path.to_string_lossy().is_empty() {
            return Err(anyhow::anyhow!("Clinical path cannot be empty"));
        }
        
        Ok(())
    }
}

impl PreprocessingConfig {
    fn validate(&self) -> Result<()> {
        if self.missing_value_strategy.is_empty() {
            return Err(anyhow::anyhow!("Missing value strategy cannot be empty"));
        }
        
        if self.feature_scaling_strategy.is_empty() {
            return Err(anyhow::anyhow!("Feature scaling strategy cannot be empty"));
        }
        
        Ok(())
    }
}

impl DataStorageConfig {
    fn validate(&self) -> Result<()> {
        if self.storage_type.is_empty() {
            return Err(anyhow::anyhow!("Storage type cannot be empty"));
        }
        
        if self.storage_type == "local" && self.local_path.is_none() {
            return Err(anyhow::anyhow!("Local path must be specified for local storage"));
        }
        
        if self.storage_type == "database" && self.database_url.is_none() {
            return Err(anyhow::anyhow!("Database URL must be specified for database storage"));
        }
        
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_config_creation() {
        let config = Config::new();
        assert_eq!(config.app.name, "AI Toolset for IPF and Aging Research");
        assert_eq!(config.app.version, "1.0.0");
    }
    
    #[test]
    fn test_config_validation() {
        let config = Config::new();
        assert!(config.validate().is_ok());
    }
    
    #[test]
    fn test_aging_clock_config_validation() {
        let mut config = AgingClockConfig::default();
        assert!(config.validate().is_ok());
        
        config.input_dim = 0;
        assert!(config.validate().is_err());
    }
    
    #[test]
    fn test_ipf_p3gpt_config_validation() {
        let mut config = IpfP3GptConfig::default();
        assert!(config.validate().is_ok());
        
        config.embedding_dim = 7;
        config.num_heads = 2;
        assert!(config.validate().is_err());
    }
    
    #[tokio::test]
    async fn test_config_file_operations() {
        let config = Config::new();
        let temp_path = "/tmp/test_config.json";
        
        // Test saving
        assert!(config.save_to_file(temp_path).await.is_ok());
        
        // Test loading
        let loaded_config = Config::from_file(temp_path).await.unwrap();
        assert_eq!(loaded_config.app.name, config.app.name);
        
        // Cleanup
        let _ = std::fs::remove_file(temp_path);
    }
}
