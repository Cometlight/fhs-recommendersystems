
# CF Recommender  (updated: 11.11.2016)

no_recommendations <- c(1, 5, 10, 20, 50, 75, 100, 200, 300)
cf_MAP <- c(15.98, 13.16, 11.63, 9.81, 7.38, 6.36, 5.66, 4.14, 3.38)
cf_MAR <- c(0.70, 2.75, 4.79, 7.89, 14.38, 18.31, 21.55, 30.94, 37.40)
cf_F1 <- c(1.34, 4.56, 6.78, 8.75, 9.75, 9.44, 8.97, 7.31, 6.20)

# CF - MAP
plot(no_recommendations, cf_MAP, type="b", xaxt="n", yaxt="n", xlab="NO_RECOMMENDED_ARTISTS", ylab="MAP", col="red", ylim=c(0,16), main="CF Recommender - MAP")
axis(1, at=no_recommendations, labels=no_recommendations)
axis(2, at=seq(0,16,by=2.0))
grid()
#text(no_recommendations+20, cf_MAP, cf_MAP) # doesnt look good because of: out of bounds problems

# CF - MAR
plot(no_recommendations, cf_MAR, type="b", xaxt="n", yaxt="n", xlab="NO_RECOMMENDED_ARTISTS", ylab="MAR", col="darkgreen", ylim=c(0,40), main="CF Recommender - MAR")
axis(1, at=no_recommendations, labels=no_recommendations)
axis(2, at=seq(0,40,by=4.0))
grid()

# CF - F1
plot(no_recommendations, cf_F1, type="b", xaxt="n", yaxt="n", xlab="NO_RECOMMENDED_ARTISTS", ylab="F1", col="blue", ylim=c(0,10), main="CF Recommender - F1")
axis(1, at=no_recommendations, labels=no_recommendations)
axis(2, at=seq(0,10,by=1.0))
grid()

################################## START HYBRID (RANK BASED) ##################################

no_recommendations <- c(1, 5, 10, 20, 50, 75, 100, 200, 300)
hrb_MAP <- c(6.39, 8.42, 7.8, 7.03, 5.63, 4.98, 4.53, 3.47, 2.9)
hrb_MAR <- c(0.32, 1.96, 3.55, 6.22, 11.93, 15.47, 18.56, 27.35, 33.59)
hrb_F1 <- c(0.62, 3.18, 4.88, 6.6, 7.65, 7.53, 7.28, 6.15, 5.33)

# HYBRID (RB) - MAP
plot(no_recommendations, hrb_MAP, type="b", xaxt="n", yaxt="n", xlab="NO_RECOMMENDED_ARTISTS", ylab="MAP", col="red", ylim=c(0,16), main="HYBRID (RB) Recommender - MAP")
axis(1, at=no_recommendations, labels=no_recommendations)
axis(2, at=seq(0,16,by=2.0))
grid()

# HYBRID (RB) - MAR
plot(no_recommendations, hrb_MAR, type="b", xaxt="n", yaxt="n", xlab="NO_RECOMMENDED_ARTISTS", ylab="MAR", col="darkgreen", ylim=c(0,40), main="HYBRID (RB) Recommender - MAR")
axis(1, at=no_recommendations, labels=no_recommendations)
axis(2, at=seq(0,40,by=4.0))
grid()

# HYBRID (RB) - F1
plot(no_recommendations, hrb_F1, type="b", xaxt="n", yaxt="n", xlab="NO_RECOMMENDED_ARTISTS", ylab="F1", col="blue", ylim=c(0,10), main="HYBRID (RB) Recommender - F1")
axis(1, at=no_recommendations, labels=no_recommendations)
axis(2, at=seq(0,10,by=1.0))
grid()

################################## END HYBRID (RANK BASED) ##################################

################################## START HYBRID (SET BASED) ##################################

no_recommendations <- c(1, 5, 10, 20, 50, 75, 100, 200, 300)
hsb_MAP <- c(15.98, 8.79, 6.86, 5.91, 4.58, 3.98, 3.54, 2.64, 2.19)
hsb_MAR <- c(0.7, 1.96, 3.05, 5.12, 9.54, 12.14, 13.94, 20.26, 24.33)
hsb_F1 <- c(1.34, 3.20, 4.22, 5.49, 6.19, 6.0, 5.64, 4.67, 4.01)

# HYBRID (SB) - MAP
plot(no_recommendations, hsb_MAP, type="b", xaxt="n", yaxt="n", xlab="NO_RECOMMENDED_ARTISTS", ylab="MAP", col="red", ylim=c(0,16), main="HYBRID (SB) Recommender - MAP")
axis(1, at=no_recommendations, labels=no_recommendations)
axis(2, at=seq(0,16,by=2.0))
grid()

# HYBRID (SB) - MAR
plot(no_recommendations, hsb_MAR, type="b", xaxt="n", yaxt="n", xlab="NO_RECOMMENDED_ARTISTS", ylab="MAR", col="darkgreen", ylim=c(0,40), main="HYBRID (SB) Recommender - MAR")
axis(1, at=no_recommendations, labels=no_recommendations)
axis(2, at=seq(0,40,by=4.0))
grid()

# HYBRID (SB) - F1
plot(no_recommendations, hsb_F1, type="b", xaxt="n", yaxt="n", xlab="NO_RECOMMENDED_ARTISTS", ylab="F1", col="blue", ylim=c(0,10), main="HYBRID (RB) Recommender - F1")
axis(1, at=no_recommendations, labels=no_recommendations)
axis(2, at=seq(0,10,by=1.0))
grid()

################################## END HYBRID (SET BASED) ##################################

################################## START POPULARITY BASED ##################################

no_recommendations <- c(1, 5, 10, 20, 50, 75, 100, 200, 300)
pb_MAP <- c(4.23, 3.77, 3.15, 3.1, 3.03, 2.88, 2.7, 2.21, 1.91)
pb_MAR <- c(0.16, 0.73, 1.14, 2.23, 5.2, 7.36, 9.1, 14.9, 19.19)
pb_F1 <- c(0.31, 1.23, 1.67, 2.6, 3.82, 4.14, 4.17, 3.85, 3.47)

# Popularity-based - MAP
plot(no_recommendations, pb_MAP, type="b", xaxt="n", yaxt="n", xlab="NO_RECOMMENDED_ARTISTS", ylab="MAP", col="red", ylim=c(0,16), main="Popularity-based Recommender - MAP")
axis(1, at=no_recommendations, labels=no_recommendations)
axis(2, at=seq(0,16,by=2.0))
grid()

# Popularity-based - MAR
plot(no_recommendations, pb_MAR, type="b", xaxt="n", yaxt="n", xlab="NO_RECOMMENDED_ARTISTS", ylab="MAR", col="darkgreen", ylim=c(0,40), main="Popularity-based Recommender - MAR")
axis(1, at=no_recommendations, labels=no_recommendations)
axis(2, at=seq(0,40,by=4.0))
grid()

# Popularity-based - F1
plot(no_recommendations, pb_F1, type="b", xaxt="n", yaxt="n", xlab="NO_RECOMMENDED_ARTISTS", ylab="F1", col="blue", ylim=c(0,10), main="Popularity-based Recommender - F1")
axis(1, at=no_recommendations, labels=no_recommendations)
axis(2, at=seq(0,10,by=1.0))
grid()

################################## END POPULARITY BASED ##################################


# CB Recommender (updated: 11.11.2016)

no_recommendations <- c(1, 5, 10, 20, 50, 75, 100, 200, 300)
cb_MAP <- c(4.07, 3.75, 3.37, 2.92, 2.30, 2.01, 1.85, 1.34, 1.15)
cb_MAR <- c(1.56, 0.85, 1.45, 2.44, 4.72, 6.07, 7.34, 12.47, 15.61)
cb_F1 <- c(2.26, 1.39, 2.02, 2.66, 3.09, 3.02, 2.95, 2.42, 2.15)

# CB - MAP
plot(no_recommendations, cb_MAP, type="b", xaxt="n", yaxt="n", xlab="NO_RECOMMENDED_ARTISTS", ylab="MAP", col="red", ylim=c(0,5), main="CB Recommender - MAP")
axis(1, at=no_recommendations, labels=no_recommendations)
axis(2, at=seq(0,5,by=0.5))
grid()
#text(no_recommendations+20, cf_MAP, cf_MAP) # doesnt look good because of: out of bounds problems

# CB - MAR
plot(no_recommendations, cb_MAR, type="b", xaxt="n", yaxt="n", xlab="NO_RECOMMENDED_ARTISTS", ylab="MAR", col="darkgreen", ylim=c(0,20), main="CB Recommender - MAR")
axis(1, at=no_recommendations, labels=no_recommendations)
axis(2, at=seq(0,20,by=2.0))
grid()

# CB - F1
plot(no_recommendations, cb_F1, type="b", xaxt="n", yaxt="n", xlab="NO_RECOMMENDED_ARTISTS", ylab="F1", col="blue", ylim=c(0,4), main="CB Recommender - F1")
axis(1, at=no_recommendations, labels=no_recommendations)
axis(2, at=seq(0,4,by=0.5))
grid()





# Random Baseline (Ours, user picking)
rb_n_artists <- c(1, 5, 10, 20, 50, 75, 100, 200, 300)
rb_MAP <- c(0.96, 1.01, 0.98, 0.95, 0.99, 1.02, 1.00, 0.96, 0.92)
rb_MAR <- c(0.04, 0.18, 0.33, 0.7, 1.73, 2.73, 3.51, 6.57, 9.40)
rb_F1 <- c(0.07, 0.3, 0.5, 0.81, 1.26, 1.48, 1.56, 1.68, 1.67)

plot(rb_n_artists, rb_MAP, type="b", xaxt="n", yaxt="n", xlab="number of artists predicted", ylab="[%]", col="red", ylim=c(0,10), main="Random Baseline (random user picking)")
par(new=TRUE)
plot(rb_n_artists, rb_MAR, type="b", xaxt="n", yaxt="n", xlab="", ylab="", col="blue", ylim=c(0,10))
par(new=TRUE)
plot(rb_n_artists, rb_F1, type="b", xaxt="n", yaxt="n", xlab="", ylab="", col="darkgreen", ylim=c(0,10))
axis(1, at=rb_n_artists, labels=rb_n_artists)
axis(2, at=seq(0,15,by=2.5))
grid()
legend(2, 9, c("MAP", "MAR", "F1 Score"), lty=c(1,1), lwd=2, col=c("red", "blue", "darkgreen"))
par(new=FALSE)

# Random Baseline (his, completely random)

rb2_n_artists <- c(1, 5, 10, 20, 50, 75, 100, 200, 300)
rb2_MAP <- c(0.36, 0.36, 0.35, 0.35, 0.37, 0.37, 0.37, 0.32, 0.32)
rb2_MAR <- c(0.01, 0.06, 0.5, 0.2, 0.11, 0.78, 1.03, 1.99, 3.03)
rb2_F1 <- c(0.02, 0.1, 0.41, 0.26, 0.16, 0.5, 0.54, 0.55, 0.58)

plot(rb2_n_artists, rb2_MAP, type="b", xaxt="n", yaxt="n", xlab="number of artists predicted", ylab="[%]", col="red", ylim=c(0,3.5), main="Random Baseline")
par(new=TRUE)
plot(rb2_n_artists, rb2_MAR, type="b", xaxt="n", yaxt="n", xlab="", ylab="", col="blue", ylim=c(0,3.5))
par(new=TRUE)
plot(rb2_n_artists, rb2_F1, type="b", xaxt="n", yaxt="n", xlab="", ylab="", col="darkgreen", ylim=c(0,3.5))
axis(1, at=rb2_n_artists, labels=rb2_n_artists)
axis(2, at=seq(0,3.5,by=0.25))
grid()
legend(20, 3, c("MAP", "MAR", "F1 Score"), lty=c(1,1), lwd=2, col=c("red", "blue", "darkgreen"))
par(new=FALSE)


# Compare results: MAP/MAR
plot(cf_MAR, cf_MAP, type="l", xaxt="n", yaxt="n", xlab="MAR [%]", ylab="MAP [%]", col="red", ylim=c(0,16), xlim=c(0,40), main="Comparison of different Recommender Systems")
points(cf_MAR, cf_MAP, pch=20, col="red")
par(new=TRUE)
plot(rb_MAR, rb_MAP, type="l", xaxt="n", yaxt="n", xlab="", ylab="", col="blue", ylim=c(0,16), xlim=c(0,40))
points(rb_MAR, rb_MAP, pch=20, col="blue")
par(new=TRUE)
plot(rb2_MAR, rb2_MAP, type="l", xaxt="n", yaxt="n", xlab="", ylab="", col="darkgreen", ylim=c(0,16), xlim=c(0,40))
points(rb2_MAR, rb2_MAP, pch=20, col="darkgreen")
par(new=TRUE)
plot(hrb_MAR, hrb_MAP, type="l", xaxt="n", yaxt="n", xlab="", ylab="", col="orange", ylim=c(0,16), xlim=c(0,40))
points(hrb_MAR, hrb_MAP, pch=20, col="orange")
par(new=TRUE)
plot(sort(cb_MAR), cb_MAP[order(cb_MAR)], type="l", xaxt="n", yaxt="n", xlab="", ylab="", col="magenta", ylim=c(0,16), xlim=c(0,40))
points(sort(cb_MAR), cb_MAP[order(cb_MAR)], pch=20, col="magenta")
axis(1, at=seq(0,40, by=5))
axis(2, at=seq(0, 16, by=1))
grid()
legend(15, 15, c("Collaborative Filtering", "Hybrid (CF-CB, rank-based)", "Content Based", "Random Baseline (user picking)", "Random Baseline"), lty=c(1,1), lwd=2, col=c("red", "orange", "magenta", "blue", "darkgreen"))
par(new=FALSE)

# Compare results: F1/number of recommendations
plot(no_recommendations, seq(0, 15, length.out=(length(no_recommendations))), type="n", xlab="Number of Recommendations", ylab="F1 Score", main="Comparison of different Recommender Systems")
lines(no_recommendations, rb_F1, col="blue")
lines(no_recommendations, rb2_F1, col="darkgreen")
lines(no_recommendations, cf_F1, col="red")
lines(no_recommendations, cb_F1, col="orange")
lines(no_recommendations, hrb_F1, col="magenta")
lines(c(50, 50), c(0,15), col="black", lty="dashed")
text(65, 14.5, "peak")
grid()
legend(140, 15.2, c("Collaborative Filtering", "Hybrid (CF-CB, rank-based)", "Content Based", "Random Baseline (user picking)", "Random Baseline"), lty=c(1,1), lwd=2, col=c("red", "magenta", "orange", "blue", "darkgreen"))
par(new=FALSE)


#  For presentation
# Random Baseline vs Random baseline with user picking
plot(rb_MAR, rb_MAP, type="l", xaxt="n", yaxt="n", xlab="MAR [%]", ylab="MAP [%]", col="blue", ylim=c(0,1.2), xlim=c(0,10))
points(rb_MAR, rb_MAP, pch=20, col="blue")
par(new=TRUE)
plot(sort(rb2_MAR), rb2_MAP[order(rb2_MAR)], type="l", xaxt="n", yaxt="n", xlab="", ylab="", col="darkgreen", ylim=c(0,1.2), xlim=c(0,10))
points(sort(rb2_MAR), rb2_MAP[order(rb2_MAR)], pch=20, col="darkgreen")
axis(1, at=seq(0,10, by=1))
axis(2, at=seq(0,1.2, by=0.2))
grid()
legend(5, 0.8, c("Random Baseline (user picking)", "Random Baseline"), lty=c(1,1), lwd=2, col=c("blue", "darkgreen"))
