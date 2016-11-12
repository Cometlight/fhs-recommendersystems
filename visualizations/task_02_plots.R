
# CF Recommender  (updated: 11.11.2016)

no_recommendations <- c(1, 5, 10, 20, 50, 75, 100, 200, 300)
cf_MAP <- c(15.98, 13.16, 11.63, 9.81, 7.38, 6.36, 5.66, 4.14, 3.38)
cf_MAR <- c(0.70, 2.75, 4.79, 7.89, 14.38, 18.31, 21.55, 30.94, 37.40)
cf_F1 <- c(1.34, 4.56, 6.78, 8.75, 9.75, 9.44, 8.97, 7.31, 6.20)

# CF - MAP
plot(no_recommendations, cf_MAP, type="b", xaxt="n", yaxt="n", xlab="k neighbours", ylab="MAP", col="red", ylim=c(0,16), main="CF Recommender - MAP")
axis(1, at=no_recommendations, labels=no_recommendations)
axis(2, at=seq(0,16,by=2.0))
grid()
#text(no_recommendations+20, cf_MAP, cf_MAP) # doesnt look good because of: out of bounds problems

# CF - MAR
plot(no_recommendations, cf_MAR, type="b", xaxt="n", yaxt="n", xlab="k neighbours", ylab="MAR", col="darkgreen", ylim=c(0,40), main="CF Recommender - MAR")
axis(1, at=no_recommendations, labels=no_recommendations)
axis(2, at=seq(0,40,by=4.0))
grid()

# CF - F1
plot(no_recommendations, cf_F1, type="b", xaxt="n", yaxt="n", xlab="k neighbours", ylab="F1", col="blue", ylim=c(0,10), main="CF Recommender - F1")
axis(1, at=no_recommendations, labels=no_recommendations)
axis(2, at=seq(0,10,by=1.0))
grid()


# CB Recommender (updated: 11.11.2016)

no_recommendations <- c(1, 5, 10, 20, 50, 75, 100, 200, 300)
cb_MAP <- c(4.07, 3.75, 3.37, 2.92, 2.30, 2.01, 1.85, 1.34, 1.15)
cb_MAR <- c(1.56, 0.85, 1.45, 2.44, 4.72, 6.07, 7.34, 12.47, 15.61)
cb_F1 <- c(2.26, 1.39, 2.02, 2.66, 3.09, 3.02, 2.95, 2.42, 2.15)







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


# Compare results
plot(cf_MAR, cf_MAP, type="b", xaxt="n", yaxt="n", xlab="MAR [%]", ylab="MAP [%]", col="red", ylim=c(0,15), xlim=c(0,3), main="Comparison of different Recommender Systems")
par(new=TRUE)
plot(rb_MAR, rb_MAP, type="b", xaxt="n", yaxt="n", xlab="", ylab="", col="blue", ylim=c(0,15), xlim=c(0,3))
par(new=TRUE)
plot(rb2_MAR, rb2_MAP, type="b", xaxt="n", yaxt="n", xlab="", ylab="", col="darkgreen", ylim=c(0,15), xlim=c(0,3))
axis(1, at=seq(0, 3, 0.2))
axis(2, at=seq(0,15,by=2.5))
grid()
legend(1.85, 9, c("Collaborative Filtering", "Random Baseline (user picking)", "Random Baseline"), lty=c(1,1), lwd=2, col=c("red", "blue", "darkgreen"))
