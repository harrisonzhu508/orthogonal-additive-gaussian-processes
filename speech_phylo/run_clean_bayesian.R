#!/usr/bin/env Rscript

# export_results_to_json.R
#
# Usage:
#   Rscript speech_phylo/export_results_to_json.R

# ─── 0. Setup ──────────────────────────────────────────────────────
suppressPackageStartupMessages({
  library(ape)
  library(brms)
  library(dplyr)
  library(tibble)
  library(jsonlite)
  library(sf)
})
# ensure output dir exists
out_dir <- "speech_phylo/results_clean2_bayesian"
if (!dir.exists(out_dir)) dir.create(out_dir, recursive = TRUE)

# ─── 1. Load BEAST helper & data ─────────────────────────────────
source("speech_phylo/beast.R")  # must define read.annot.beast() & extract_beast_metadata()

tree_names <- c("heggarty2024", "long_v2_CCD0.0.05", "long_v3_CCD0.0.10", "long_v3_CCD0.0.25")

for (tree_name in tree_names) {

    tree_path     <- paste0("speech_phylo/", tree_name, ".nex")  # BEAST .trees or .nexus file
    glotto_path   <- "speech_phylo/glottolog_df.csv"

    tr            <- read.annot.beast(tree_path)
    glottolog_df  <- read.csv(glotto_path, stringsAsFactors = FALSE)

    # ─── 2. Extract & merge metadata ─────────────────────────────────
    metadata_df <- extract_beast_metadata(tree = tr)
    df <- inner_join(metadata_df, glottolog_df, by = "language") %>%
        column_to_rownames(var = "language")

    # ─── 3. Prune tree & compute cophenetic distances ────────────────
    to_drop <- setdiff(tr$tip.label, rownames(df))
    tr      <- drop.tip(tr, to_drop)
    coph_df <- as.data.frame(cophenetic(unroot(tr)))

    # 4) compute raw phylogenetic covariance (shared branch lengths)
    V_raw <- vcv(unroot(tr))   # n×n matrix

    # 5) write metadata.csv (one row per tip, with language column)
    meta_out_df <- df %>% rownames_to_column("language")
    write.csv(
        meta_out_df,
        file = file.path(out_dir, paste0("metadata_", tree_name, ".csv")),
        row.names = FALSE,
        quote = TRUE
    )

    # 6) write V_raw.csv (with row & col names)
    write.csv(V_raw, file= file.path(out_dir, paste0("V_raw_", tree_name, ".csv")),
                row.names = TRUE, col.names = TRUE, quote = FALSE)

    cat("Wrote:\n",
        " • metadata →", file.path(out_dir, paste0("metadata_", tree_name, ".csv")), "\n",
        " • raw phylo covariance →", file.path(out_dir, paste0("V_raw_", tree_name, ".csv")))

    # ─── 4. Fit models with/without phylogenetic covariance ───────────────────
    for (add_phylocov in c(TRUE, FALSE)) {
        for (mercator in c(FALSE, TRUE)) {
            # Initialize scaling parameters
            x_center <- NULL
            x_scale <- NULL
            y_center <- NULL
            y_scale <- NULL
            log_n_center <- NULL
            log_n_scale <- NULL
            
            # Store original values for later reference
            df$longitude_orig <- df$longitude
            df$latitude_orig <- df$latitude
            df$log_n_speakers_orig <- log(df$n_speakers)
            
            # Normalize log(n_speakers) - this is done for both mercator and non-mercator
            log_n_scaled <- scale(log(df$n_speakers))
            df$log_n_speakers_norm <- log_n_scaled[, 1]
            log_n_center <- attr(log_n_scaled, "scaled:center")
            log_n_scale <- attr(log_n_scaled, "scaled:scale")
            
            # Prepare coordinates based on mercator flag
            if (mercator) {
                # Create sf object with lat/lon coordinates (EPSG:4326)
                sf_points <- st_as_sf(df, 
                                    coords = c("longitude_orig", "latitude_orig"), 
                                    crs = 4326,
                                    remove = FALSE)  # Keep original columns
                
                # Transform to Web Mercator (EPSG:3857)
                sf_points_proj <- st_transform(sf_points, crs = 3857)
                
                # Extract the projected coordinates
                coords_proj <- st_coordinates(sf_points_proj)
                
                # Store original (unscaled) coordinates for reference
                df$x_meters <- coords_proj[, "X"]
                df$y_meters <- coords_proj[, "Y"]
                
                # Normalize the Mercator coordinates
                x_scaled <- scale(coords_proj[, "X"])
                y_scaled <- scale(coords_proj[, "Y"])
                
                df$x_norm <- x_scaled[, 1]  # Normalized X
                df$y_norm <- y_scaled[, 1]  # Normalized Y
                
                # Store scaling parameters for later use
                x_center <- attr(x_scaled, "scaled:center")
                x_scale <- attr(x_scaled, "scaled:scale")
                y_center <- attr(y_scaled, "scaled:center")
                y_scale <- attr(y_scaled, "scaled:scale")
                
                coord_vars <- c("x_norm", "y_norm")
            } else {
                # Normalize latitude and longitude
                lon_scaled <- scale(df$longitude)
                lat_scaled <- scale(df$latitude)
                
                df$longitude_norm <- lon_scaled[, 1]
                df$latitude_norm <- lat_scaled[, 1]
                
                # Store scaling parameters
                x_center <- attr(lon_scaled, "scaled:center")  # longitude center
                x_scale <- attr(lon_scaled, "scaled:scale")    # longitude scale
                y_center <- attr(lat_scaled, "scaled:center")  # latitude center
                y_scale <- attr(lat_scaled, "scaled:scale")    # latitude scale
                
                coord_vars <- c("longitude_norm", "latitude_norm")
            }
            
            for (interaction in c(FALSE, TRUE)) {
                # Build formula dynamically based on coordinate system
                # Now using normalized variables
                if (interaction) {
                    if (mercator) {
                        formula_obj <- log(rate_median) ~ x_norm + y_norm + log_n_speakers_norm + 
                                       x_norm*y_norm + x_norm*log_n_speakers_norm + y_norm*log_n_speakers_norm
                    } else {
                        formula_obj <- log(rate_median) ~ latitude_norm + longitude_norm + log_n_speakers_norm + 
                                       longitude_norm*latitude_norm + latitude_norm*log_n_speakers_norm + longitude_norm*log_n_speakers_norm
                    }
                } else {
                    if (mercator) {
                        formula_obj <- log(rate_median) ~ x_norm + y_norm + log_n_speakers_norm
                    } else {
                        formula_obj <- log(rate_median) ~ longitude_norm + latitude_norm + log_n_speakers_norm
                    }
                }
                
                # Fit Bayesian regression (with optional phylogenetic covariance)
                model_df <- df
                model_df$log_rate_median <- log(model_df$rate_median)
                language_levels <- rownames(model_df)
                model_df$language_factor <- factor(language_levels, levels = language_levels)

                predictors <- c(coord_vars, "log_n_speakers_norm")
                if (interaction) {
                    if (mercator) {
                        predictors <- c(predictors,
                                        "x_norm:y_norm",
                                        "x_norm:log_n_speakers_norm",
                                        "y_norm:log_n_speakers_norm")
                    } else {
                        predictors <- c(predictors,
                                        "longitude_norm:latitude_norm",
                                        "latitude_norm:log_n_speakers_norm",
                                        "longitude_norm:log_n_speakers_norm")
                    }
                }

                formula_str <- paste("log_rate_median ~", paste(unique(predictors), collapse = " + "))

                brm_data2 <- NULL
                if (add_phylocov) {
                    V <- V_raw[language_levels, language_levels]
                    formula_str <- paste(formula_str, "+ (1|gr(language_factor, cov = V))")
                    brm_data2 <- list(V = V)
                }

                formula_obj <- as.formula(formula_str)

                brm_args <- list(
                    formula = formula_obj,
                    data = model_df,
                    family = gaussian(),
                    refresh = 0,
                    chains = 4,
                    iter = 4000,
                    warmup = 1000,
                    seed = 20231103,
                    cores = max(1L, min(4L, parallel::detectCores())),
                    control = list(adapt_delta = 0.99, max_treedepth = 15)
                )
                if (!is.null(brm_data2)) {
                    brm_args$data2 <- brm_data2
                }

                model <- do.call(brm, brm_args)
                model_type <- if (add_phylocov) "brms_phylo" else "brms"
                lambda_hat <- NA_real_

                cat(sprintf("\nPhylocov: %s, Mercator: %s, Interaction: %s\n", 
                            add_phylocov, mercator, interaction))

                coef_df <- as.data.frame(brms::fixef(model, probs = c(0.025, 0.975)))
                colnames(coef_df) <- c("Estimate", "Est.Error", "CI_lower", "CI_upper")

                cat("\nPosterior summaries (normalized scale):\n")
                print(coef_df[, c("Estimate", "Est.Error", "CI_lower", "CI_upper")])

                # ─── Calculate unstandardized coefficients for interpretability ─────
                unstd_coefs <- data.frame(
                    term = rownames(coef_df),
                    estimate_normalized = coef_df[, "Estimate"],
                    se_normalized = coef_df[, "Est.Error"],
                    lower_normalized = coef_df[, "CI_lower"],
                    upper_normalized = coef_df[, "CI_upper"],
                    estimate_original = NA_real_,
                    se_original = NA_real_,
                    lower_original = NA_real_,
                    upper_original = NA_real_,
                    estimate_per_1000km = NA_real_,
                    se_per_1000km = NA_real_,
                    lower_per_1000km = NA_real_,
                    upper_per_1000km = NA_real_,
                    stringsAsFactors = FALSE
                )
                rownames(unstd_coefs) <- NULL

                for (i in seq_len(nrow(unstd_coefs))) {
                    term <- unstd_coefs$term[i]

                    if (term == "(Intercept)") {
                        unstd_coefs$estimate_original[i] <- unstd_coefs$estimate_normalized[i]
                        unstd_coefs$se_original[i] <- unstd_coefs$se_normalized[i]
                        unstd_coefs$lower_original[i] <- unstd_coefs$lower_normalized[i]
                        unstd_coefs$upper_original[i] <- unstd_coefs$upper_normalized[i]

                    } else if (mercator) {
                        if (term == "x_norm") {
                            scale_factor <- x_scale
                            unstd_coefs$estimate_original[i] <- unstd_coefs$estimate_normalized[i] / scale_factor
                            unstd_coefs$se_original[i] <- unstd_coefs$se_normalized[i] / scale_factor
                            unstd_coefs$lower_original[i] <- unstd_coefs$lower_normalized[i] / scale_factor
                            unstd_coefs$upper_original[i] <- unstd_coefs$upper_normalized[i] / scale_factor
                            unstd_coefs$estimate_per_1000km[i] <- unstd_coefs$estimate_original[i] * 1e6
                            unstd_coefs$se_per_1000km[i] <- unstd_coefs$se_original[i] * 1e6
                            unstd_coefs$lower_per_1000km[i] <- unstd_coefs$lower_original[i] * 1e6
                            unstd_coefs$upper_per_1000km[i] <- unstd_coefs$upper_original[i] * 1e6

                        } else if (term == "y_norm") {
                            scale_factor <- y_scale
                            unstd_coefs$estimate_original[i] <- unstd_coefs$estimate_normalized[i] / scale_factor
                            unstd_coefs$se_original[i] <- unstd_coefs$se_normalized[i] / scale_factor
                            unstd_coefs$lower_original[i] <- unstd_coefs$lower_normalized[i] / scale_factor
                            unstd_coefs$upper_original[i] <- unstd_coefs$upper_normalized[i] / scale_factor
                            unstd_coefs$estimate_per_1000km[i] <- unstd_coefs$estimate_original[i] * 1e6
                            unstd_coefs$se_per_1000km[i] <- unstd_coefs$se_original[i] * 1e6
                            unstd_coefs$lower_per_1000km[i] <- unstd_coefs$lower_original[i] * 1e6
                            unstd_coefs$upper_per_1000km[i] <- unstd_coefs$upper_original[i] * 1e6

                        } else if (term == "log_n_speakers_norm") {
                            scale_factor <- log_n_scale
                            unstd_coefs$estimate_original[i] <- unstd_coefs$estimate_normalized[i] / scale_factor
                            unstd_coefs$se_original[i] <- unstd_coefs$se_normalized[i] / scale_factor
                            unstd_coefs$lower_original[i] <- unstd_coefs$lower_normalized[i] / scale_factor
                            unstd_coefs$upper_original[i] <- unstd_coefs$upper_normalized[i] / scale_factor

                        } else if (term == "x_norm:y_norm") {
                            scale_factor <- x_scale * y_scale
                            unstd_coefs$estimate_original[i] <- unstd_coefs$estimate_normalized[i] / scale_factor
                            unstd_coefs$se_original[i] <- unstd_coefs$se_normalized[i] / scale_factor
                            unstd_coefs$lower_original[i] <- unstd_coefs$lower_normalized[i] / scale_factor
                            unstd_coefs$upper_original[i] <- unstd_coefs$upper_normalized[i] / scale_factor

                        } else if (term == "x_norm:log_n_speakers_norm") {
                            scale_factor <- x_scale * log_n_scale
                            unstd_coefs$estimate_original[i] <- unstd_coefs$estimate_normalized[i] / scale_factor
                            unstd_coefs$se_original[i] <- unstd_coefs$se_normalized[i] / scale_factor
                            unstd_coefs$lower_original[i] <- unstd_coefs$lower_normalized[i] / scale_factor
                            unstd_coefs$upper_original[i] <- unstd_coefs$upper_normalized[i] / scale_factor
                            unstd_coefs$estimate_per_1000km[i] <- unstd_coefs$estimate_original[i] * 1e6
                            unstd_coefs$se_per_1000km[i] <- unstd_coefs$se_original[i] * 1e6
                            unstd_coefs$lower_per_1000km[i] <- unstd_coefs$lower_original[i] * 1e6
                            unstd_coefs$upper_per_1000km[i] <- unstd_coefs$upper_original[i] * 1e6

                        } else if (term == "y_norm:log_n_speakers_norm") {
                            scale_factor <- y_scale * log_n_scale
                            unstd_coefs$estimate_original[i] <- unstd_coefs$estimate_normalized[i] / scale_factor
                            unstd_coefs$se_original[i] <- unstd_coefs$se_normalized[i] / scale_factor
                            unstd_coefs$lower_original[i] <- unstd_coefs$lower_normalized[i] / scale_factor
                            unstd_coefs$upper_original[i] <- unstd_coefs$upper_normalized[i] / scale_factor
                            unstd_coefs$estimate_per_1000km[i] <- unstd_coefs$estimate_original[i] * 1e6
                            unstd_coefs$se_per_1000km[i] <- unstd_coefs$se_original[i] * 1e6
                            unstd_coefs$lower_per_1000km[i] <- unstd_coefs$lower_original[i] * 1e6
                            unstd_coefs$upper_per_1000km[i] <- unstd_coefs$upper_original[i] * 1e6
                        }
                    } else {
                        if (term == "longitude_norm") {
                            scale_factor <- x_scale
                            unstd_coefs$estimate_original[i] <- unstd_coefs$estimate_normalized[i] / scale_factor
                            unstd_coefs$se_original[i] <- unstd_coefs$se_normalized[i] / scale_factor
                            unstd_coefs$lower_original[i] <- unstd_coefs$lower_normalized[i] / scale_factor
                            unstd_coefs$upper_original[i] <- unstd_coefs$upper_normalized[i] / scale_factor

                        } else if (term == "latitude_norm") {
                            scale_factor <- y_scale
                            unstd_coefs$estimate_original[i] <- unstd_coefs$estimate_normalized[i] / scale_factor
                            unstd_coefs$se_original[i] <- unstd_coefs$se_normalized[i] / scale_factor
                            unstd_coefs$lower_original[i] <- unstd_coefs$lower_normalized[i] / scale_factor
                            unstd_coefs$upper_original[i] <- unstd_coefs$upper_normalized[i] / scale_factor

                        } else if (term == "log_n_speakers_norm") {
                            scale_factor <- log_n_scale
                            unstd_coefs$estimate_original[i] <- unstd_coefs$estimate_normalized[i] / scale_factor
                            unstd_coefs$se_original[i] <- unstd_coefs$se_normalized[i] / scale_factor
                            unstd_coefs$lower_original[i] <- unstd_coefs$lower_normalized[i] / scale_factor
                            unstd_coefs$upper_original[i] <- unstd_coefs$upper_normalized[i] / scale_factor

                        } else if (term == "longitude_norm:latitude_norm") {
                            scale_factor <- x_scale * y_scale
                            unstd_coefs$estimate_original[i] <- unstd_coefs$estimate_normalized[i] / scale_factor
                            unstd_coefs$se_original[i] <- unstd_coefs$se_normalized[i] / scale_factor
                            unstd_coefs$lower_original[i] <- unstd_coefs$lower_normalized[i] / scale_factor
                            unstd_coefs$upper_original[i] <- unstd_coefs$upper_normalized[i] / scale_factor

                        } else if (term == "longitude_norm:log_n_speakers_norm") {
                            scale_factor <- x_scale * log_n_scale
                            unstd_coefs$estimate_original[i] <- unstd_coefs$estimate_normalized[i] / scale_factor
                            unstd_coefs$se_original[i] <- unstd_coefs$se_normalized[i] / scale_factor
                            unstd_coefs$lower_original[i] <- unstd_coefs$lower_normalized[i] / scale_factor
                            unstd_coefs$upper_original[i] <- unstd_coefs$upper_normalized[i] / scale_factor

                        } else if (term == "latitude_norm:log_n_speakers_norm") {
                            scale_factor <- y_scale * log_n_scale
                            unstd_coefs$estimate_original[i] <- unstd_coefs$estimate_normalized[i] / scale_factor
                            unstd_coefs$se_original[i] <- unstd_coefs$se_normalized[i] / scale_factor
                            unstd_coefs$lower_original[i] <- unstd_coefs$lower_normalized[i] / scale_factor
                            unstd_coefs$upper_original[i] <- unstd_coefs$upper_normalized[i] / scale_factor
                        }
                    }
                }

                r2_vals <- as.numeric(brms::bayes_R2(model))
                r_squared_summary <- list(
                    mean = mean(r2_vals),
                    sd = stats::sd(r2_vals),
                    q2.5 = unname(stats::quantile(r2_vals, 0.025)),
                    q97.5 = unname(stats::quantile(r2_vals, 0.975))
                )

                fitted_summary <- fitted(model, summary = TRUE, probs = c(0.025, 0.975))
                preds_df <- data.frame(
                    language = language_levels,
                    fitted_mean = fitted_summary[, "Estimate"],
                    fitted_sd = fitted_summary[, "Est.Error"],
                    fitted_lower = fitted_summary[, "Q2.5"],
                    fitted_upper = fitted_summary[, "Q97.5"],
                    stringsAsFactors = FALSE
                )

                # ─── 5. Collect results into a list ──────────────────────────────
                results_list <- list(
                    metadata = as.data.frame(df) %>% 
                                rownames_to_column("language"),
                    coph       = if(add_phylocov) coph_df else NULL,
                    add_phylocov = add_phylocov,
                    mercator   = mercator,
                    interaction = interaction,
                    model_type = model_type,
                    call       = formula_str,
                    lambda_hat = lambda_hat,
                    r_squared  = r_squared_summary,
                    summary    = coef_df %>%
                                rownames_to_column("term"),
                    preds      = preds_df,
                    scaling_params = list(
                        x_center = x_center,
                        x_scale = x_scale,
                        y_center = y_center,
                        y_scale = y_scale,
                        log_n_center = log_n_center,
                        log_n_scale = log_n_scale
                    ),
                    unstandardized_coefficients = unstd_coefs
                )

                # ─── 6. Write JSON ───────────────────────────────────────────────
                json_path <- file.path(out_dir, 
                                    paste0("phylolm_results_", tree_name, 
                                            "_mercator", mercator,
                                            "_interaction", interaction,
                                            "_add_phylocov", add_phylocov, ".json"))
                write_json(
                    results_list,
                    path       = json_path,
                    pretty     = TRUE,
                    auto_unbox = TRUE
                )

                cat("✔️ Results written to", json_path, "\n")

                # ─── 7. Save results as CSV ─────────────────────────────────────
                block_latlon <- FALSE

                # Create results dataframe for CSV
                csv_results <- data.frame(
                    model = model_type,
                    tree = tree_name,
                    mercator = mercator,
                    interaction = interaction,
                    block_latlon = block_latlon,
                    add_phylocov = add_phylocov,
                    lambda = ifelse(is.na(lambda_hat), NA, lambda_hat)
                )

                # Add coefficient information
                for(i in seq_len(nrow(coef_df))) {
                    # Clean term names for CSV column names
                    term_name <- rownames(coef_df)[i]
                    term_name_clean <- gsub("_norm", "", term_name)  # Remove _norm suffix
                    term_name_clean <- gsub(":", "_", term_name_clean)  # Replace : with _

                    # Add normalized coefficients (these are what the model actually uses)
                    csv_results[paste0("coef_", term_name_clean)] <- coef_df[i, "Estimate"]
                    csv_results[paste0("se_", term_name_clean)] <- coef_df[i, "Est.Error"]
                    csv_results[paste0("ci_lower_", term_name_clean)] <- coef_df[i, "CI_lower"]
                    csv_results[paste0("ci_upper_", term_name_clean)] <- coef_df[i, "CI_upper"]

                    csv_results[paste0("pval_", term_name_clean)] <- NA_real_

                    # Add unstandardized coefficients for interpretability
                    if (!is.null(unstd_coefs)) {
                        row_idx <- which(unstd_coefs$term == rownames(coef_df)[i])
                        if (length(row_idx) > 0) {
                            if (mercator) {
                                csv_results[paste0("coef_meters_", term_name_clean)] <- unstd_coefs$estimate_original[row_idx]
                                csv_results[paste0("se_meters_", term_name_clean)] <- unstd_coefs$se_original[row_idx]
                                csv_results[paste0("ci_lower_meters_", term_name_clean)] <- unstd_coefs$lower_original[row_idx]
                                csv_results[paste0("ci_upper_meters_", term_name_clean)] <- unstd_coefs$upper_original[row_idx]
                                if (!is.na(unstd_coefs$estimate_per_1000km[row_idx])) {
                                    csv_results[paste0("coef_per1000km_", term_name_clean)] <- unstd_coefs$estimate_per_1000km[row_idx]
                                    csv_results[paste0("se_per1000km_", term_name_clean)] <- unstd_coefs$se_per_1000km[row_idx]
                                    csv_results[paste0("ci_lower_per1000km_", term_name_clean)] <- unstd_coefs$lower_per_1000km[row_idx]
                                    csv_results[paste0("ci_upper_per1000km_", term_name_clean)] <- unstd_coefs$upper_per_1000km[row_idx]
                                }
                            } else {
                                csv_results[paste0("coef_original_", term_name_clean)] <- unstd_coefs$estimate_original[row_idx]
                                csv_results[paste0("se_original_", term_name_clean)] <- unstd_coefs$se_original[row_idx]
                                csv_results[paste0("ci_lower_original_", term_name_clean)] <- unstd_coefs$lower_original[row_idx]
                                csv_results[paste0("ci_upper_original_", term_name_clean)] <- unstd_coefs$upper_original[row_idx]
                            }
                        }
                    }
                }

                # Add all scaling parameters
                csv_results$x_scale_sd <- x_scale
                csv_results$y_scale_sd <- y_scale
                csv_results$x_center <- x_center
                csv_results$y_center <- y_center
                csv_results$log_n_scale_sd <- log_n_scale
                csv_results$log_n_center <- log_n_center

                # Save CSV
                csv_path <- file.path("speech_phylo/results_clean2_bayesian",
                                    paste0("phylolm_", tree_name,
                                        "_mercator", mercator,
                                        "_interaction", interaction,
                                        "_blocklatlon", block_latlon,
                                        "_add_phylocov", add_phylocov, ".csv"))

                write.csv(csv_results, csv_path, row.names = FALSE)
                cat("✔️ CSV results written to", csv_path, "\n")
            }
        }
    }
}